import gc
from pathlib import Path
from datetime import datetime
import sys
import argparse
from functools import partial
from operator import attrgetter

try:
    sys.path.append(str(Path(".").resolve()))
except:
    raise RuntimeError("Can't append root directory of the project to the path")

from comet_ml.integration.pytorch import log_model, watch
from omegaconf import OmegaConf, ListConfig
from rich import print
import numpy as np
from tqdm import tqdm
from icecream import ic, install
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torcheval.metrics import PeakSignalNoiseRatio
import matplotlib
import matplotlib.pyplot as plt
from einops import rearrange

from model.vertiencoder import VertiEncoder, load_model

# from model.swae import SWAutoencoder, Encoder, Decoder
from model.dataloader import VertiEncoderDataset
from utils.nn import op_counter, init_optimizer
from utils.io import save_checkpoint, load_checkpoint
from utils.helpers import (
    get_conf,
    timeit,
    init_logger,
    init_device,
    to_tensor,
    fix_seed,
    get_exp_number,
)
from utils.loss import sliced_wasserstein_distance


class Learner:
    def __init__(self, cfg_dir: str):
        # load config file
        self.cfg = get_conf(cfg_dir)
        # set the name for the model
        exp_number = get_exp_number(
            self.cfg.directory.save, self.cfg.logger.experiment_name
        )
        self.cfg.directory.model_name = f"{self.cfg.logger.experiment_name}-{exp_number}-{self.cfg.model.transformer_layer.d_model}D"
        self.cfg.directory.model_name += f"-{datetime.now():%m-%d-%H-%M}"
        self.cfg.logger.experiment_name = self.cfg.directory.model_name
        self.cfg.directory.save = str(
            Path(self.cfg.directory.save) / self.cfg.directory.model_name
        )
        # if debugging True, set a few rules
        if self.cfg.train_params.debug:
            install()
            ic.enable()
            ic.configureOutput(prefix=lambda: f"{datetime.now():%H:%M:%S} |> ")
            torch.autograd.set_detect_anomaly(True)
            self.cfg.logger.disabled = True
            matplotlib.use("TkAgg")
            self.cfg.train_params.compile = False
        else:
            matplotlib.use("Agg")
            ic.disable()
            torch.autograd.set_detect_anomaly(True)
        # initialize the logger and the device
        self.logger = init_logger(self.cfg)
        self.logger.disable_mp()
        self.device = init_device(self.cfg)
        # fix the seed for reproducibility
        fix_seed(self.cfg.train_params.seed)
        torch.backends.cudnn.benchmark = True
        # torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision("high")
        # creating dataset interface and dataloader for trained data
        self.data, self.val_data = self.init_dataloader()
        # create model and initialize its weights and move them to the device
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - INITIALIZING the model!")
        self.model = load_model(self.cfg.model)
        self.model = self.model.to(self.device)
        self.block_size = self.model.block_size
        print(
            f"{datetime.now():%Y-%m-%d %H:%M:%S} - Model block size: {self.block_size}"
        )
        if self.cfg.train_params.compile:
            self.uncompiled_model = self.model
            self.model = torch.compile(self.model)
        # log the model gradients, weights, and activations in comet
        watch(self.model)
        self.logger.log_code(folder="./vertiformer/model")
        # initialize the optimizer
        self.optimizer, self.scheduler = init_optimizer(
            self.cfg, self.model.parameters(), self.cfg.train_params.optimizer
        )
        num_params = [x.numel() for x in self.model.parameters()]
        trainable_params = [
            x.numel() for x in self.model.parameters() if x.requires_grad
        ]
        print(
            f"{datetime.now():%Y-%m-%d %H:%M:%S} - Number of parameters: {sum(num_params) / 1e6:.2f}M"
        )
        print(
            f"{datetime.now():%Y-%m-%d %H:%M:%S} - Number of trainable parameters: {sum(trainable_params) / 1e6:.2f}M"
        )
        # define loss function
        self.criterion = torch.nn.MSELoss()
        self.psnr = PeakSignalNoiseRatio()
        self.optimizer.zero_grad(set_to_none=True)
        # if resuming, load the checkpoint
        self.if_resume()

    def train(self):
        """Trains the model"""

        while self.epoch <= self.cfg.train_params.epochs:
            running_loss = []
            running_loss_poses = []
            running_loss_actions = []

            bar = tqdm(
                self.data,
                desc=f"Epoch {self.epoch:03}/{self.cfg.train_params.epochs:03}, training: ",
                bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
            )
            for data in bar:
                self.iteration += 1
                (loss_data), t_train = self.forward_batch(data)
                t_train /= self.data.batch_size
                running_loss.append(loss_data["loss"])

                bar.set_postfix(
                    loss=loss_data["loss"],
                    loss_label=loss_data["loss_label"],
                    Grad_Norm=loss_data["grad_norm"],
                    Time=t_train,
                )

                self.logger.log_metrics(
                    {
                        "batch_loss": loss_data["loss"],
                        "loss_label": loss_data["loss_label"],
                        "grad_norm": loss_data["grad_norm"],
                    },
                    epoch=self.epoch,
                    step=self.iteration,
                )
                self.logger.log_image(
                    loss_data["samples"],
                    f"train_E{self.epoch}",
                    step=self.iteration,
                    image_channels="first",
                )

            bar.close()
            self.scheduler.step()

            # validate on val set
            (val_loss), t = self.validate()
            t /= len(self.val_data.dataset)

            # average loss for an epoch
            self.e_loss.append(np.mean(running_loss))  # epoch loss
            print(
                f"{datetime.now():%Y-%m-%d %H:%M:%S} Epoch {self.epoch:03}, "
                + f"Iteration {self.iteration:05} summary: train Loss: "
                + f"[green]{self.e_loss[-1]:.4f}[/green] \t| Val loss: [red]{val_loss:.4f}[/red] \t| "
                + f"PSNR: [red]{self.psnr.compute().item():.3f}[/red] "
                + f"\t| time: {t:.6f} seconds\n"
            )

            self.logger.log_metrics(
                {
                    "train_loss": self.e_loss[-1],
                    "val_loss": val_loss,
                    "PSNR": self.psnr.compute().item(),
                    "time": t,
                },
                epoch=self.epoch,
                step=self.iteration,
            )

            if self.epoch % self.cfg.train_params.save_every == 0 or (
                self.psnr.compute().item() > self.best
                and self.epoch >= self.cfg.train_params.start_saving_best
            ):
                self.save()

            gc.collect()
            torch.cuda.empty_cache()
            self.epoch += 1
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Training is DONE!")

    @timeit(attrgetter("device"))
    def forward_batch(self, data):
        """Forward pass of a batch"""
        self.model.train()
        # move data to device
        patch, next_patch, cmd_vel, pose, labels, mask = data
        patch = patch.to(self.device).squeeze(2)
        next_patch = next_patch.to(self.device).squeeze(2)
        cmd_vel = cmd_vel.to(self.device)  # (B, T, L) -> (B, L, T, 1); L = 2
        pose = pose.to(self.device)  #  (B, T, L) -> (B, L, T, 1); L = 6
        labels = labels.to(self.device).unsqueeze(1)

        B = pose.size(0)
        # forward, backward
        (
            _,
            pred_labels,
            pred_next_patch,
            pred_patch,
            pred_actions_token,
            pred_pose_token,
        ) = self.model(patch.clone(), cmd_vel.clone(), pose.clone(), mask)

        patch_gt = patch.clone()
        if self.cfg.train_params.norm_pix_loss:
            # based on MAE, predicting the normalized ground truth helps the model learn better
            mean = next_patch.mean(dim=-1, keepdim=True)
            var = next_patch.var(dim=-1, keepdim=True)
            next_patch = (next_patch - mean) / (var + 1.0e-6) ** 0.5
            # also for masked patches
            mean = patch_gt.mean(dim=-1, keepdim=True)
            var = patch_gt.var(dim=-1, keepdim=True)
            patch_gt = (patch_gt - mean) / (var + 1.0e-6) ** 0.5

        loss = self.criterion(pred_next_patch, next_patch)
        loss_label = F.binary_cross_entropy_with_logits(pred_labels, labels)
        loss = loss + loss_label
        loss = loss + self.criterion(
            pred_patch[torch.arange(B).unsqueeze(1), mask["patches"], :],
            patch_gt[torch.arange(B).unsqueeze(1), mask["patches"], :],
        )
        loss = loss + self.criterion(
            pred_actions_token[torch.arange(B).unsqueeze(1), mask["action"], :],
            cmd_vel[torch.arange(B).unsqueeze(1), mask["action"], :],
        )
        loss = loss + self.criterion(
            pred_pose_token[torch.arange(B).unsqueeze(1), mask["pose"], :],
            pose[torch.arange(B).unsqueeze(1), mask["pose"], :],
        )

        loss = loss / self.cfg.train_params.accumulation_steps
        loss.backward()
        # gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.train_params.grad_clipping
        )
        # update
        if self.iteration % self.cfg.train_params.accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        recon_samples_hat = pred_next_patch[:64].detach().cpu().unsqueeze(2)
        log_samples = make_grid(
            [
                make_grid(
                    [gt[i], pred[i]],
                    nrow=2,
                    value_range=(-1, 1),
                    normalize=True,
                    scale_each=True,
                )
                for gt, pred in zip(
                    next_patch[:64].unsqueeze(2).cpu(), recon_samples_hat
                )  # loop over the batch
                for i in range(gt.size(1))  # loop over time
            ],
            nrow=8,
            padding=5,
        )

        return {
            "loss": loss.item(),
            "loss_label": loss_label.item(),
            "grad_norm": grad_norm.item(),
            "samples": log_samples,
        }

    @timeit(attrgetter("device"))
    @torch.no_grad()
    def validate(self):

        self.model.eval()

        running_loss = []
        self.psnr.reset()
        bar = tqdm(
            self.val_data,
            desc=f"Epoch {self.epoch:03}/{self.cfg.train_params.epochs:03}, validating",
            bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
        )
        for data in bar:
            # move data to device
            patch, next_patch, cmd_vel, pose = data
            patch = patch.to(self.device).squeeze(2)  # (B, T, H, W)
            next_patch = next_patch.to(self.device).squeeze(2)
            cmd_vel = cmd_vel.to(self.device)
            pose = pose.to(self.device)
            # forward, backward
            pred_patch = self.model(patch, cmd_vel, pose)
            pred_patch = self.model.img_decoder(pred_patch).view(next_patch.shape)

            if self.cfg.train_params.norm_pix_loss:
                # based on MAE, predicting the normalized ground truth helps the model learn better
                mean = next_patch.mean(dim=-1, keepdim=True)
                var = next_patch.var(dim=-1, keepdim=True)
                next_patch = (next_patch - mean) / (var + 1.0e-6) ** 0.5

            loss = self.criterion(pred_patch, next_patch)

            recon_samples_hat = pred_patch.detach().cpu()
            next_patch = next_patch.cpu()
            self.psnr.update(recon_samples_hat, next_patch)

            log_patch_samples = make_grid(
                [
                    make_grid(
                        [gt[i], pred[i]],
                        nrow=2,
                        value_range=(-1, 1),
                        normalize=True,
                        scale_each=True,
                    )
                    for gt, pred in zip(
                        next_patch[:64].cpu().unsqueeze(2),
                        recon_samples_hat.unsqueeze(2),
                    )  # loop over the batch
                    for i in range(gt.size(1))  # loop over time
                ],
                nrow=8,
                padding=5,
            )

            running_loss.append(loss.item())
            bar.set_postfix(loss=loss.item(), PSNR=self.psnr.compute().item())
            self.logger.log_image(
                log_patch_samples,
                f"val_E{self.epoch}",
                step=self.iteration,
                image_channels="first",
            )
        bar.close()
        # average loss
        loss = np.mean(running_loss)
        return loss

    def init_dataloader(self):
        """Initializes the dataloaders"""
        print(
            f"{datetime.now():%Y-%m-%d %H:%M:%S} - INITIALIZING the train and val dataloaders!"
        )
        # defining the dataset interface
        dataset = VertiEncoderDataset(**self.cfg.dataset)
        self.cfg.dataset.update(self.cfg.val_dataset)
        val_dataset = VertiEncoderDataset(**self.cfg.dataset)
        # creating dataloader
        data = DataLoader(dataset, **self.cfg.dataloader)

        self.cfg.dataloader.update({"shuffle": False})  # for val dataloader
        val_data = DataLoader(val_dataset, **self.cfg.dataloader)

        # log dataset status
        self.logger.log_parameters(
            {"train_len": len(dataset), "val_len": len(val_dataset)}
        )
        print(
            f"Training consists of {len(dataset)} samples, and validation consists of {len(val_dataset)} samples."
        )

        return data, val_data

    def if_resume(self):
        if self.cfg.logger.resume:
            # load checkpoint
            print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - LOADING checkpoint!!!")
            save_dir = self.cfg.directory.load
            checkpoint = load_checkpoint(save_dir, self.device)
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["lr_scheduler"])
            self.epoch = checkpoint["epoch"] + 1
            self.e_loss = checkpoint["e_loss"]
            self.iteration = checkpoint["iteration"] + 1
            self.best = checkpoint["best"]
            print(
                f"{datetime.now():%Y-%m-%d %H:%M:%S} "
                + f"LOADING checkpoint was successful, start from epoch {self.epoch}"
                + f" and loss {self.best}"
            )
        else:
            self.epoch = 1
            self.iteration = 0
            self.best = -np.inf
            self.e_loss = []

        self.logger.set_epoch(self.epoch)

    def save(self, name=None):
        model = self.uncompiled_model if self.cfg.train_params.compile else self.model

        checkpoint = {
            "time": str(datetime.now()),
            "epoch": self.epoch,
            "iteration": self.iteration,
            "model": model.state_dict(),
            "model_name": type(model).__name__,
            "optimizer": self.optimizer.state_dict(),
            "optimizer_name": type(self.optimizer).__name__,
            "lr_scheduler": self.scheduler.state_dict(),
            "best": self.best,
            "e_loss": self.e_loss,
        }

        if name is None:
            save_name = f"{self.cfg.directory.model_name}-E{self.epoch}"
        else:
            save_name = name

        if self.psnr.compute().item() > self.best:
            self.best = self.psnr.compute().item()
            checkpoint["best"] = self.best
            save_checkpoint(checkpoint, True, self.cfg.directory.save, save_name)
        else:
            save_checkpoint(checkpoint, False, self.cfg.directory.save, save_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", default="conf/vertiformer", type=str)
    args = parser.parse_args()
    cfg_path = args.conf
    learner = Learner(cfg_path)
    learner.train()
