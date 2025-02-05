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
from omegaconf import DictConfig

from model.vertiencoder import load_model
from model.dt_models import FKD, BehaviorCloning, IKD
from model.dataloader import VertiEncoderDownStream
from utils.nn import check_grad_norm, init_weights, op_counter, init_optimizer
from utils.io import save_checkpoint, load_checkpoint
from utils.helpers import get_conf, timeit, init_logger, init_device, fix_seed
from utils.visualize import visualize_trajectory, visualize_traj_diff
from utils.hist_loss import HistogramLoss


class Learner:
    def __init__(self, cfg: DictConfig):
        # load config file
        self.cfg = cfg
        # set the name for the model
        pretext_exp = Path(self.cfg.model.transformer_weight).stem[4:6]
        self.cfg.logger.experiment_name = f"{self.cfg.dataset.task.upper()}{pretext_exp}-PL{self.cfg.dataset.pred_len}-{'finetuned' if self.cfg.model.finetune else 'frozen'}"
        self.cfg.directory.model_name = (
            f"{self.cfg.logger.experiment_name}-{datetime.now():%m-%d-%H-%M}"
        )
        self.cfg.logger.experiment_name = self.cfg.directory.model_name
        self.cfg.directory.save = str(
            Path(self.cfg.directory.save) / self.cfg.directory.model_name
        )
        print(
            f"{datetime.now():%Y-%m-%d %H:%M:%S} - Starting experiment {self.cfg.logger.experiment_name}!"
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
            ic.disable()
            matplotlib.use("Agg")
            torch.autograd.set_detect_anomaly(True)
        # initialize the logger and the device
        self.logger = init_logger(self.cfg)
        self.device = init_device(self.cfg)
        fix_seed(self.cfg.train_params.seed)
        torch.backends.cudnn.benchmark = True
        # torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision("high")
        # creating dataset interface and dataloader for trained data
        self.data, self.val_data, self.stats = self.init_dataloader()
        self.stats["pose_diff_mean"] = self.stats["pose_diff_mean"].to(self.device)
        self.stats["pose_diff_std"] = self.stats["pose_diff_std"].to(self.device)
        # create model and initialize its weights and move them to the device
        self.pretext_model, self.dt_model = self.init_model(self.cfg.dataset.task)
        if self.cfg.train_params.compile:
            self.pretext_model = torch.compile(self.pretext_model)
            self.dt_model = torch.compile(self.dt_model)
        # log the model gradients, weights, and activations in comet
        # watch(self.dt_model)
        self.logger.log_code(folder="./vertiformer/model/")
        # initialize the optimizer
        self.optimizer, self.scheduler = init_optimizer(
            self.cfg,
            list(self.dt_model.parameters()) + list(self.pretext_model.parameters()),
            self.cfg.train_params.optimizer,
        )
        num_params = [
            x.numel()
            for x in list(self.dt_model.parameters())
            + list(self.pretext_model.parameters())
        ]
        trainable_params = [
            x.numel()
            for x in list(self.dt_model.parameters())
            + list(self.pretext_model.parameters())
            if x.requires_grad
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
        self.scale = self.stats["pose_diff_std"].unsqueeze(0)
        # if resuming, load the checkpoint
        self.if_resume()

    def train(self):
        """Trains the model"""
        while self.epoch <= self.cfg.train_params.epochs:
            running_loss = []

            bar = tqdm(
                self.data,
                desc=f"Epoch {self.epoch:03}/{self.cfg.train_params.epochs:03}, training: ",
                bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
            )
            for data in bar:
                self.iteration += 1
                (loss, grad_norm, error_rate, action_loss), t_train = (
                    self.forward_batch(data)
                )
                t_train /= self.data.batch_size
                running_loss.append(loss.item())
                if error_rate is not None:
                    error_rate = error_rate.mean(dim=0)

                bar.set_postfix(
                    loss=loss.item(),
                    Grad_Norm=grad_norm,
                    ActionLoss=action_loss,
                    Time=t_train,
                )

                self.logger.log_metrics(
                    {
                        "batch_loss": loss.item(),
                        "grad_norm": grad_norm,
                        # "error_rateTr": np.mean([error_rate[0], error_rate[1], error_rate[2]]),
                        "loss_actionsTr": action_loss,
                    },
                    epoch=self.epoch,
                    step=self.iteration,
                )

            bar.close()
            self.scheduler.step()

            # validate on val set
            (val_loss, error_rate, loss_action), t = self.validate()
            t /= len(self.val_data.dataset)
            self.val_loss = val_loss
            if error_rate is not None:
                error_rate = error_rate.mean(dim=0)

            # average loss for an epoch
            self.e_loss.append(np.mean(running_loss))  # epoch loss
            log = (
                f"{datetime.now():%Y-%m-%d %H:%M:%S} Epoch {self.epoch:03}, "
                + f"Iteration {self.iteration:05} summary: TrainLoss: "
                + f"[green]{self.e_loss[-1]:.3f}[/green] | ValLoss: [red]{val_loss:.3f}[/red] "
                + f"| time: {t:.3f} seconds\n"
            )
            if self.cfg.dataset.task == "reconstruction":
                log = (
                    log[:-1]
                    + f"\t| PSNR: [red]{self.psnr.compute().item():.3f}[/red]\n"
                )

            print(log)

            metrics = {
                "train_loss": self.e_loss[-1],
                "val_loss": val_loss,
                "time": t,
                "loss_actionsVal": loss_action,
            }

            if loss_action:
                metrics["loss_actionsVal"] = loss_action
            if error_rate is not None:
                metrics["error_rateX"] = error_rate[0]
                metrics["error_rateY"] = error_rate[1]
                metrics["error_rateZ"] = error_rate[2]
            if self.cfg.dataset.task == "reconstruction":
                metrics["PSNR"] = self.psnr.compute().item()
            self.logger.log_metrics(
                metrics,
                epoch=self.epoch,
                step=self.iteration,
            )

            if self.epoch % self.cfg.train_params.save_every == 0 or (
                val_loss < self.best
                and self.epoch >= self.cfg.train_params.start_saving_best
            ):
                self.save()

            gc.collect()
            self.epoch += 1

        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Training is DONE!")

    @timeit(attrgetter("device"))
    def forward_batch(self, batch):
        """Forward pass of a batch"""
        self.dt_model.train()
        if self.cfg.model.finetune:
            self.pretext_model.train()
        patch = batch[0].to(self.device).squeeze(2)
        cmd_vel = batch[1].to(self.device)
        verti_pose = batch[2].to(self.device)
        gt = batch[3].to(self.device)
        # forward, backward
        pred_len = self.cfg.dataset.pred_len  # get observation length T (B, T, C).

        if self.cfg.model.finetune:
            z, _, _, _, _, _ = self.pretext_model(
                patch, cmd_vel[:, :-pred_len], verti_pose[:, :-pred_len]
            )
        else:
            z = self.pretext_model(
                patch, cmd_vel[:, :-pred_len], verti_pose[:, :-pred_len]
            )

        if self.cfg.dataset.task == "fkd":
            pred = self.dt_model(
                z, cmd_vel[:, -(pred_len + 1) : -1]
            )  # pass the current cmd_vel as well
        elif self.cfg.dataset.task == "ikd":
            pred = self.dt_model(z, verti_pose[:, -(pred_len + 1) : -1])

        else:  # BC and Reconstruction
            pred = self.dt_model(z)

        loss = F.mse_loss(pred, gt, reduction="none").mean(dim=1).mean()

        loss = loss / self.cfg.train_params.accumulation_steps
        loss.backward()
        # gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.dt_model.parameters(), self.cfg.train_params.grad_clipping
        )
        if self.cfg.model.finetune:
            torch.nn.utils.clip_grad_norm_(
                self.pretext_model.parameters(), self.cfg.train_params.grad_clipping
            )
        # update
        if self.iteration % self.cfg.train_params.accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            if self.cfg.dataset.task == "fkd":
                # |predicted - ground truth| / |ground truth|
                error_rate = torch.abs(pred[:, :, :3] - gt[:, :, :3]).mean(
                    dim=0, keepdim=True
                ) / gt[:, :, :3].abs().mean(
                    dim=0, keepdim=True
                )  # mean across batch dim
                error_rate = error_rate.squeeze().cpu()

                # random sample to visualize
                idx = torch.randint(0, pred.size(0), (1,)).item()

                fig_traj = visualize_trajectory(pred[idx].cpu(), gt[idx].cpu())
                self.logger.log_figure(
                    figure_name=f"E{self.epoch:03}|TT{self.iteration:05}",
                    figure=fig_traj,
                    step=self.iteration,
                )
                plt.close()
                loss_action = None
            else:
                loss_action = loss.item()
                error_rate = None

        return loss, grad_norm.item(), error_rate, loss_action

    @timeit(attrgetter("device"))
    @torch.no_grad()
    def validate(self):

        self.dt_model.eval()
        self.pretext_model.eval()
        running_loss_actions = []
        running_loss_poses = []
        running_error_rate = []

        running_loss = []
        bar = tqdm(
            self.val_data,
            desc=f"Epoch {self.epoch:03}/{self.cfg.train_params.epochs:03}, validating",
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        )
        for batch in bar:
            # move data to device
            patch = batch[0].to(self.device)
            cmd_vel = batch[1].to(self.device)
            verti_pose = batch[2].to(self.device)
            gt = batch[3].to(self.device)
            pred_len = self.cfg.dataset.pred_len  # get observation length T (B, T, C).

            if self.cfg.model.finetune:
                z, _, _, _, _, _ = self.pretext_model(
                    patch, cmd_vel[:, :-pred_len], verti_pose[:, :-pred_len]
                )
            else:
                z = self.pretext_model(
                    patch, cmd_vel[:, :-pred_len], verti_pose[:, :-pred_len]
                )

            if self.cfg.dataset.task == "fkd":
                pred = self.dt_model(
                    z, cmd_vel[:, -(pred_len + 1) : -1]
                )  # pass the current cmd_vel as well
            elif self.cfg.dataset.task == "ikd":
                pred = self.dt_model(z, verti_pose[:, -(pred_len + 1) : -1])
            else:  # BC and Reconstruction
                pred = self.dt_model(z)

            loss = F.mse_loss(pred, gt, reduction="none").mean(dim=1).mean()

            if self.cfg.dataset.task == "reconstruction":
                self.psnr.update(pred, gt)
                running_loss.append(loss.item())
                bar.set_postfix(loss=loss.item(), PSNR=self.psnr.compute().item())
            elif self.cfg.dataset.task == "fkd":
                # |predicted - ground truth| / |ground truth|
                error_rate = torch.abs(pred[:, :, :3] - gt[:, :, :3]).mean(
                    dim=0, keepdim=True
                ) / gt[:, :, :3].abs().mean(
                    dim=0, keepdim=True
                )  # mean across batch dim
                error_rate = error_rate.squeeze().cpu()
                running_loss.append(loss.item())
                bar.set_postfix(
                    loss=loss.item(),
                    error_rateX=error_rate[-1, 0].item(),
                    error_rateY=error_rate[-1, 1].item(),
                    error_rateZ=error_rate[-1, 2].item(),
                )

                running_error_rate.append(error_rate)
            else:
                running_loss.append(loss.item())
                bar.set_postfix(loss=loss.item())

        if self.cfg.dataset.task == "fkd":
            # random sample to visualize
            idx = torch.randint(0, pred.size(0), (1,)).item()

            fig_traj = visualize_trajectory(pred[idx].cpu(), gt[idx].cpu())
            self.logger.log_figure(
                figure_name=f"E{self.epoch:03}|VT{self.iteration:05}",
                figure=fig_traj,
                step=self.iteration,
            )
            plt.close()
            error_rate = torch.stack(running_error_rate, dim=0).mean(dim=0)
            loss_action = None
        else:
            loss_action = np.mean(running_loss).item()
            error_rate = None

        bar.close()
        # average loss
        loss = np.mean(running_loss)

        return loss, error_rate, loss_action

    def init_model(self, task: str):
        """Initializes the model"""
        print(f"{datetime.now():%H:%M:%S} - INITIALIZING the model!")
        pretext_model = load_model(self.cfg.model.transformer)
        pretext_weight = torch.load(
            self.cfg.model.transformer_weight, map_location="cpu", weights_only=False
        )["model"]
        unwanted_prefix = "_orig_mod."
        for k, v in list(pretext_weight.items()):
            if k.startswith(unwanted_prefix):
                pretext_weight[k[len(unwanted_prefix) :]] = pretext_weight.pop(k)
        pretext_model.load_state_dict(pretext_weight)
        # freeze the pretext model
        if not self.cfg.model.finetune:
            pretext_model.requires_grad_(False)
            pretext_model.eval()

        if task == "fkd":
            self.cfg.model.fkd_model.fc.dims.insert(
                0, self.cfg.model.transformer.transformer_layer.d_model
            )
            dt_model = FKD(self.cfg.model.fkd_model)

        elif task == "bc":
            self.cfg.model.bc_model.dims.insert(
                0, self.cfg.model.transformer.transformer_layer.d_model
            )
            dt_model = BehaviorCloning(self.cfg.model.bc_model)

        elif task == "ikd":
            self.cfg.model.ikd_model.pose.dims.insert(
                0, self.cfg.model.transformer.transformer_layer.d_model
            )
            dt_model = IKD(self.cfg.model.ikd_model)

        else:
            raise Exception(f"{task} is not a valid task!")

        pretext_model = pretext_model.to(device=self.device)
        dt_model = dt_model.to(device=self.device)
        return pretext_model, dt_model

    def init_dataloader(self):
        """Initializes the dataloaders"""
        print(
            f"{datetime.now():%H:%M:%S} - Training [green]{self.cfg.dataset.task.upper()}[/green] task!"
        )
        print(
            f"{datetime.now():%H:%M:%S} - INITIALIZING the train and val dataloaders!"
        )
        # defining the dataset interface
        dataset = VertiEncoderDownStream(**self.cfg.dataset)
        self.cfg.dataset.update(self.cfg.val_dataset)
        val_dataset = VertiEncoderDownStream(**self.cfg.dataset)
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

        return data, val_data, dataset.stats

    def if_resume(self):
        if self.cfg.logger.resume:
            # load checkpoint
            print(f"{datetime.now():%H:%M:%S} - LOADING checkpoint!!!")
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
                f"{datetime.now():%H:%M:%S} "
                + f"LOADING checkpoint was successful, start from epoch {self.epoch}"
                + f" and loss {self.best}"
            )
        else:
            self.epoch = 1
            self.iteration = 0
            self.best = np.inf
            self.e_loss = []

        self.logger.set_epoch(self.epoch)

    def save(self, name=None):
        checkpoint = {
            "time": str(datetime.now()),
            "epoch": self.epoch,
            "iteration": self.iteration,
            "dt_model": self.dt_model.state_dict(),
            "pretext_model": self.pretext_model.state_dict(),
            "task": self.cfg.dataset.task,
            "model_name": type(self.dt_model).__name__,
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

        if self.val_loss < self.best:
            self.best = self.val_loss
            checkpoint["best"] = self.best
            save_checkpoint(checkpoint, True, self.cfg.directory.save, save_name)
            if self.cfg.logger.upload_model:
                # upload only the current checkpoint
                log_model(self.logger, checkpoint, model_name=save_name)
        else:
            save_checkpoint(checkpoint, False, self.cfg.directory.save, save_name)
            if self.cfg.logger.upload_model:
                # upload only the current checkpoint
                log_model(self.logger, checkpoint, model_name=save_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", default="conf/dt", type=str)
    args = parser.parse_args()
    cfg_path = args.conf
    cfg: DictConfig = get_conf(cfg_path)
    learner = Learner(cfg)
    learner.train()
