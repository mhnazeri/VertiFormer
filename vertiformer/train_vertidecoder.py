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

from comet_ml.integration.pytorch import watch
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

from model.vertidecoder import load_model
from model.dataloader import VertiDecoderDataset
from utils.io import save_checkpoint, load_checkpoint
from utils.visualize import visualize_trajectory
from utils.helpers import (
    get_conf,
    timeit,
    init_logger,
    init_device,
    to_tensor,
    fix_seed,
    get_exp_number,
)


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
        print(
            f"{datetime.now():%Y-%m-%d %H:%M:%S} - INITIALIZING the model for experiment {self.cfg.logger.experiment_name}!"
        )
        self.model, self.optimizer, self.scheduler = load_model(self.cfg)
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
                bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
            )
            for data in bar:
                self.iteration += 1
                (loss_data), t_train = self.forward_batch(data)
                t_train /= self.data.batch_size
                running_loss.append(loss_data["loss"])
                running_loss_poses.append(loss_data["loss_poses"])
                running_loss_actions.append(loss_data["loss_actions"])
                loss_data["error_rate"] = loss_data["error_rate"].mean(dim=0)

                bar.set_postfix(
                    loss=loss_data["loss"],
                    lossPoses=loss_data["loss_poses"],
                    lossActions=loss_data["loss_actions"],
                    Grad_Norm=loss_data["grad_norm"],
                    Time=t_train,
                    error_rateX=loss_data["error_rate"][0].item(),
                    error_rateY=loss_data["error_rate"][1].item(),
                    error_rateZ=loss_data["error_rate"][2].item(),
                )

                self.logger.log_metrics(
                    {
                        "batch_loss": loss_data["loss"],
                        "loss_posesTr": loss_data["loss_poses"],
                        "loss_actionsTr": loss_data["loss_actions"],
                        "error_rateTr": np.mean(
                            [
                                loss_data["error_rate"][0],
                                loss_data["error_rate"][1],
                                loss_data["error_rate"][2],
                            ]
                        ),
                        "grad_norm": loss_data["grad_norm"],
                    },
                    epoch=self.epoch,
                    step=self.iteration,
                )

            bar.close()
            self.scheduler.step()

            # validate on val set
            (val_loss, loss_poses, loss_actions, error_rate), t = self.validate()
            t /= len(self.val_data.dataset)
            self.val_loss.append(val_loss)
            error_rate = error_rate.mean(dim=0)

            # average loss for an epoch
            self.e_loss.append(np.mean(running_loss))  # epoch loss
            print(
                f"{datetime.now():%Y-%m-%d %H:%M:%S} Epoch {self.epoch:03}, "
                + f"Iteration {self.iteration:05} summary: train Loss: "
                + f"[green]{self.e_loss[-1]:.4f}[/green] | Val loss: [red]{val_loss:.4f}[/red] | "
                + f"LossActions: [red]{loss_actions:.4f}[/red], LossPoses: [red]{loss_poses:.4f}[/red] | "
                + f"ErrorRateX: [red]{error_rate[0]:.3f}[/red], ErrorRateY: [red]{error_rate[1]:.3f}[/red], ErrorRateZ: [red]{error_rate[2]:.3f}[/red] | "
                + f"PSNR: [red]{self.psnr.compute().item():.3f}[/red] | "
                + f"Time: {t:.6f} seconds\n"
            )

            self.logger.log_metrics(
                {
                    "train_loss": self.e_loss[-1],
                    "val_loss": val_loss,
                    "loss_posesVal": loss_poses,
                    "loss_actionsVal": loss_actions,
                    "error_rateX": error_rate[0],
                    "error_rateY": error_rate[1],
                    "error_rateZ": error_rate[2],
                    "PSNR": self.psnr.compute().item(),
                    "time": t,
                },
                epoch=self.epoch,
                step=self.iteration,
            )

            if self.epoch % self.cfg.train_params.save_every == 0 or (
                self.val_loss[-1].item() < self.best
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
        patch, actions, poses, future_patch, future_actions, future_poses = data
        patch = patch.to(self.device).squeeze(2)
        actions = actions.to(self.device)
        poses = poses.to(self.device)
        future_patch = future_patch.to(self.device).squeeze(2)
        future_actions = future_actions.to(self.device)
        future_poses = future_poses.to(self.device)
        B = poses.size(0)
        # forward, backward
        pred_poses, pred_actions, pred_images = self.model(patch, actions, poses)

        loss_poses = (
            F.mse_loss(pred_poses, future_poses, reduction="none").mean(dim=1).mean()
        )  # temporal mean first
        loss_actions = (
            F.mse_loss(pred_actions, future_actions, reduction="none")
            .mean(dim=1)
            .mean()
        )
        patch_gt = future_patch.clone()
        if self.cfg.train_params.norm_pix_loss:
            # based on MAE, predicting the normalized ground truth helps the model learn better
            mean = patch_gt.mean(dim=-1, keepdim=True)
            var = patch_gt.var(dim=-1, keepdim=True)
            patch_gt = (patch_gt - mean) / (var + 1.0e-6) ** 0.5

        loss_patch = (
            F.mse_loss(pred_images, patch_gt, reduction="none").mean(dim=1).mean()
        )

        loss = loss_poses + loss_actions + loss_patch
        loss = loss / self.cfg.train_params.accumulation_steps  # scaling down the loss
        loss.backward()
        # gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.train_params.grad_clipping
        )
        # update
        if self.iteration % self.cfg.train_params.accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            error_rate = torch.abs(pred_poses[:, :, :3] - future_poses[:, :, :3]).mean(
                dim=0, keepdim=True
            ) / future_poses[:, :, :3].abs().mean(
                dim=0, keepdim=True
            )  # mean across batch dim
            error_rate = error_rate.squeeze().cpu()
            # visualize a random pose
            idx = torch.randint(0, B, (1,))
            fig_traj = visualize_trajectory(
                pred_poses[idx], future_poses[idx].cpu(), orientation=False
            )
            self.logger.log_figure(
                figure_name=f"E{self.epoch:03}|T{self.iteration:05}",
                figure=fig_traj,
                step=self.iteration,
            )
            plt.close()

            recon_samples_hat = pred_images[:64].detach().cpu().unsqueeze(2)
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
                        patch_gt[:64].unsqueeze(2).cpu(), recon_samples_hat
                    )  # loop over the batch
                    for i in range(gt.size(1))  # loop over time
                ],
                nrow=8,
                padding=5,
            )

            self.logger.log_image(
                log_samples,
                f"Tr_E{self.epoch}|{self.iteration:05}",
                step=self.iteration,
                image_channels="first",
            )

        return {
            "loss_poses": loss_poses.item(),
            "loss_actions": loss_actions.item(),
            "loss": loss.item(),
            "error_rate": error_rate,
            "grad_norm": grad_norm.item(),
        }

    @timeit(attrgetter("device"))
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        self.psnr.reset()
        running_loss = []
        running_loss_actions = []
        running_loss_poses = []
        running_error_rate = []
        bar = tqdm(
            self.val_data,
            desc=f"Epoch {self.epoch:03}/{self.cfg.train_params.epochs:03}, validating",
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        )
        for data in bar:
            # move data to device
            patch, actions, poses, future_patch, future_actions, future_poses = data
            patch = patch.to(self.device).squeeze(2)
            actions = actions.to(self.device)
            poses = poses.to(self.device)
            future_patch = future_patch.to(self.device).squeeze(2)
            future_actions = future_actions.to(self.device)
            future_poses = future_poses.to(self.device)
            B = poses.size(0)
            # forward
            pred_poses, pred_actions, pred_images = self.model.generate(
                patch, actions, poses, steps=self.cfg.model.pred_len
            )

            loss_poses = (
                F.mse_loss(pred_poses, future_poses, reduction="none")
                .mean(dim=1)
                .mean()
            )  # temporal mean first
            loss_actions = (
                F.mse_loss(pred_actions, future_actions, reduction="none")
                .mean(dim=1)
                .mean()
            )

            patch_gt = future_patch.clone()
            if self.cfg.train_params.norm_pix_loss:
                # based on MAE, predicting the normalized ground truth helps the model learn better
                mean = patch_gt.mean(dim=-1, keepdim=True)
                var = patch_gt.var(dim=-1, keepdim=True)
                patch_gt = (patch_gt - mean) / (var + 1.0e-6) ** 0.5

            loss_patch = (
                F.mse_loss(pred_images, patch_gt, reduction="none").mean(dim=1).mean()
            )
            self.psnr.update(pred_images, patch_gt)
            loss = loss_poses + loss_actions + loss_patch

            error_rate = torch.abs(pred_poses[:, :, :3] - future_poses[:, :, :3]).mean(
                dim=0, keepdim=True
            ) / future_poses[:, :, :3].abs().mean(
                dim=0, keepdim=True
            )  # mean across batch dim
            error_rate = error_rate.squeeze().cpu()
            # visualize a random pose
            idx = torch.randint(0, B, (1,))
            fig_traj = visualize_trajectory(
                pred_poses[idx], future_poses[idx].cpu(), orientation=False
            )
            self.logger.log_figure(
                figure_name=f"E{self.epoch:03}|T{self.iteration:05}",
                figure=fig_traj,
                step=self.iteration,
            )

            plt.close()

            recon_samples_hat = pred_images[:64].detach().cpu().unsqueeze(2)
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
                        patch_gt[:64].unsqueeze(2).cpu(), recon_samples_hat
                    )  # loop over the batch
                    for i in range(gt.size(1))  # loop over time
                ],
                nrow=8,
                padding=5,
            )

            self.logger.log_image(
                log_samples,
                f"val_E{self.epoch}|{self.iteration:05}",
                step=self.iteration,
                image_channels="first",
            )

            running_loss.append(loss.item())
            running_loss_poses.append(loss_poses.item())
            running_loss_actions.append(loss_actions.item())
            running_error_rate.append(error_rate)
            bar.set_postfix(
                loss=loss.item(),
                error_rateX=error_rate[-1, 0].item(),
                error_rateY=error_rate[-1, 1].item(),
                error_rateZ=error_rate[-1, 2].item(),
            )
        bar.close()
        # average loss
        loss = np.mean(running_loss)
        loss_actions = np.mean(running_loss_actions)
        loss_poses = np.mean(running_loss_poses)
        error_rate = torch.stack(running_error_rate, dim=0).mean(dim=0)
        return loss, loss_poses, loss_actions, error_rate

    def init_dataloader(self):
        """Initializes the dataloaders"""
        print(
            f"{datetime.now():%Y-%m-%d %H:%M:%S} - INITIALIZING the train and val dataloaders!"
        )
        # defining the dataset interface
        dataset = VertiDecoderDataset(**self.cfg.dataset)
        self.cfg.dataset.update(self.cfg.val_dataset)
        val_dataset = VertiDecoderDataset(**self.cfg.dataset)
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
            self.best = np.inf
            self.e_loss = []
            self.val_loss = []

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

        if self.val_loss[-1].item() < self.best:
            self.best = self.val_loss[-1].item()
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
