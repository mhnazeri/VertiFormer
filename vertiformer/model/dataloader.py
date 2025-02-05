from pathlib import Path
import pickle
from collections import defaultdict
from itertools import compress

import numpy as np
import torch
from torchvision.transforms import v2
from torch.utils.data import Dataset
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from vertiformer.utils.helpers import to_tensor, read_patch


class TvertiDatasetBase(Dataset):

    def __init__(
        self,
        root: str,
        stats: str,
        train: bool = True,
        obs_len: int = 20,
        f_size: int = 7,
        height_diff: int = 0.5,
        pred_len: int = 1,
        resize: int = 14,
    ):

        self.obs_len = obs_len
        self.height_diff = height_diff
        self.root = Path(root)
        self.pred_len = pred_len
        self.train = train
        with open(self.root, "rb") as f:
            metadata = pickle.load(f)

        with open(stats, "rb") as f:
            self.stats = pickle.load(f)

        self.metadata = defaultdict(list)
        self.f_size = f_size
        for i, bag_name in enumerate(metadata["bag_name"]):
            if len(metadata["data"][i]["cmd_vel"]) < self.f_size:
                continue
            num_samples = len(metadata["data"][i]["cmd_vel"])
            bag_data = defaultdict(list)
            cmd_vel = np.array(metadata["data"][i]["cmd_vel"], dtype=np.float32)
            cmd_vel_filtered = np.zeros(
                (cmd_vel.shape[0] - self.f_size + 1, cmd_vel.shape[1]), dtype=np.float32
            )
            cmd_vel_filtered[:, 0] = np.convolve(
                cmd_vel[:, 0], np.ones(self.f_size) / self.f_size, mode="valid"
            )
            cmd_vel_filtered[:, 1] = np.convolve(
                cmd_vel[:, 1], np.ones(self.f_size) / self.f_size, mode="valid"
            )
            cmd_vel = cmd_vel_filtered

            pose_diff = np.array(metadata["data"][i]["pose_diff"], dtype=np.float32)
            pose_diff_filtered = np.zeros(
                (pose_diff.shape[0] - self.f_size + 1, pose_diff.shape[1]),
                dtype=np.float32,
            )
            for k in range(6):
                pose_diff_filtered[:, k] = np.convolve(
                    pose_diff[:, k], np.ones(self.f_size) / self.f_size, mode="valid"
                )
            pose_diff = pose_diff_filtered / self.stats["pose_diff_max"]
            # trimming data because of convolution
            metadata["data"][i]["footprint"] = metadata["data"][i]["footprint"][
                self.f_size // 2 : -self.f_size // 2
            ]
            metadata["data"][i]["pose"] = metadata["data"][i]["pose"][
                self.f_size // 2 : -self.f_size // 2
            ]
            metadata["data"][i]["dt"] = metadata["data"][i]["dt"][
                self.f_size // 2 : -self.f_size // 2
            ]
            metadata["data"][i]["time"] = metadata["data"][i]["time"][
                self.f_size // 2 : -self.f_size // 2
            ]
            metadata["data"][i]["motor_speed"] = metadata["data"][i]["motor_speed"][
                self.f_size // 2 : -self.f_size // 2
            ]
            num_samples = num_samples - (self.f_size - 1)

            for j in range(num_samples - self.obs_len - self.pred_len):
                bag_data["cmd_vel"].append(
                    cmd_vel[j : j + self.obs_len + self.pred_len].tolist()
                )
                bag_data["footprint"].append(
                    metadata["data"][i]["footprint"][
                        j : j + self.obs_len + self.pred_len
                    ]
                )
                bag_data["pose"].append(
                    metadata["data"][i]["pose"][j : j + self.obs_len + self.pred_len]
                )
                bag_data["motor_speed"].append(
                    metadata["data"][i]["motor_speed"][
                        j : j + self.obs_len + self.pred_len
                    ]
                )
                bag_data["dt"].append(
                    metadata["data"][i]["dt"][j : j + self.obs_len + self.pred_len]
                )
                bag_data["pose_diff"].append(
                    pose_diff[j : j + self.obs_len + self.pred_len].tolist()
                )
                bag_data["time"].append(
                    metadata["data"][i]["time"][j : j + self.obs_len + self.pred_len]
                )

            self.metadata["cmd_vel"].extend(bag_data["cmd_vel"])
            self.metadata["footprint"].extend(bag_data["footprint"])
            self.metadata["pose"].extend(bag_data["pose"])
            self.metadata["motor_speed"].extend(bag_data["motor_speed"])
            self.metadata["dt"].extend(bag_data["dt"])
            self.metadata["pose_diff"].extend(bag_data["pose_diff"])
            self.metadata["time"].extend(bag_data["time"])

        self.transform = v2.Compose(
            [
                v2.Resize(size=(resize, resize), antialias=True),
                v2.ToDtype(torch.float32, scale=False),
            ]
        )

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class TvertiDatasetAENextToken(TvertiDatasetBase):
    def __init__(
        self,
        root: str,
        stats: str,
        train: bool = True,
        f_size: int = 7,
        height_diff: int = 0.5,
    ):
        super().__init__(
            root=root, stats=stats, train=train, f_size=f_size, height_diff=height_diff
        )

    def __len__(self):
        return len(self.metadata["cmd_vel"]) - self.block_size - 16

    def __getitem__(self, idx):
        """Return a sample in the form: (patch, next_patch)"""
        patch = self.transform(
            read_patch(
                self.root.parents[0] / self.metadata["footprint"][idx],
                self.metadata["pose"][idx][2],
                self.height_diff,
            )
        )
        patch = (patch - self.stats["footprint_mean"]) / self.stats["footprint_std"]
        # next patch
        next_patch = self.transform(
            read_patch(
                self.root.parents[0]
                / self.metadata["footprint"][idx + self.block_size + 15],
                self.metadata["pose"][idx + self.block_size + 15][2],
                self.height_diff,
            )
        )
        next_patch = (next_patch - self.stats["footprint_mean"]) / self.stats[
            "footprint_std"
        ]
        next_cmd_vel = torch.stack(
            [
                (to_tensor(self.metadata["cmd_vel"][i]) - self.stats["cmd_vel_mean"])
                / self.stats["cmd_vel_std"]
                for i in range(idx + self.block_size, idx + self.block_size + 15)
            ],
            dim=0,
        )
        current_cmd_vel = to_tensor(self.metadata["cmd_vel"][idx])
        return patch, next_patch, current_cmd_vel, next_cmd_vel


class VertiEncoderDataset(TvertiDatasetBase):
    def __init__(
        self,
        root: str,
        stats: str,
        train: bool = True,
        block_size: int = 6,
        data_frequency: int = 10,
        model_frequency: int = 3,
        obs_len: int = 2,
        f_size: int = 7,
        height_diff: int = 0.5,
        pred_len: int = 4,
        resize: int = 14,
        mask_percentage: float = 0.15,
    ):
        self.obs_len = obs_len * data_frequency
        super().__init__(
            root=root,
            stats=stats,
            train=train,
            f_size=f_size,
            height_diff=height_diff,
            pred_len=(pred_len * model_frequency) + 1,
            resize=resize,
            obs_len=self.obs_len,
        )
        self.data_frequency = data_frequency
        self.pred_len = pred_len
        self.block_size = block_size
        self.mask_percentage = mask_percentage
        self.model_frequency = model_frequency

    def __len__(self):
        return (
            len(self.metadata["pose"])
            - ((self.pred_len * self.model_frequency) + 1)
            - self.obs_len
        )

    def __getitem__(self, idx: int):
        # patch input to the model
        # get random indexes
        obs_indices = torch.sort(
            torch.randperm(self.obs_len)[: self.block_size]
        ).values  # tensor of indexes

        pred_indices = (
            torch.arange(
                self.model_frequency,
                (self.pred_len * self.model_frequency) + 1,
                self.model_frequency,
            )
            + self.obs_len
        )

        patch = torch.stack(
            [
                (
                    self.transform(
                        read_patch(
                            self.root.parents[0] / self.metadata["footprint"][idx][i],
                            self.metadata["pose"][idx][i][2],
                            self.height_diff,
                        )
                    ).squeeze()
                )
                for i in obs_indices
            ],
            dim=0,
        )
        # next patch the model should predict
        next_patch = torch.stack(
            [
                (
                    self.transform(
                        read_patch(
                            self.root.parents[0] / self.metadata["footprint"][idx][i],
                            self.metadata["pose"][idx][i][2],
                            self.height_diff,
                        )
                    ).squeeze()
                )
                for i in pred_indices
            ],
            dim=0,
        )

        if self.train:
            if torch.rand(1).item() > 0.25:
                scalar = patch.std(dim=(1, 2), keepdim=True) / (
                    (0.3 + torch.rand(patch.size())) * 10
                )
                msk = torch.rand(patch.size()) > 0.5
                scalar[msk] = -scalar[msk]
                patch = patch + scalar

            if torch.rand(1).item() > 0.25:
                scalar = next_patch.std(dim=(1, 2), keepdim=True) / (
                    (0.3 + torch.rand(next_patch.size())) * 10
                )
                msk = torch.rand(next_patch.size()) > 0.5
                scalar[msk] = -scalar[msk]
                next_patch = next_patch + scalar

        cmd_vel = to_tensor([self.metadata["cmd_vel"][idx][i] for i in obs_indices])
        pose = to_tensor([self.metadata["pose_diff"][idx][i] for i in obs_indices])

        if self.train:
            if torch.rand(1).item() > 0.5:
                scalar = 1 + ((torch.rand(cmd_vel.shape) / 5) - 0.1)
                cmd_vel = cmd_vel * scalar
                cmd_vel = torch.clip(cmd_vel, -1, 1)

            if torch.rand(1).item() > 0.5:
                scalar = 1 + ((torch.rand(pose.shape) / 10) - 0.05)
                pose = pose * scalar

        if self.train:
            # ic(perm)
            to_mask = int(len(obs_indices) * self.mask_percentage)
            # ic(to_mask)
            perm = torch.randperm(len(obs_indices))
            mask_patches = perm[:to_mask]
            perm = torch.randperm(len(obs_indices))  # for action
            mask_action = perm[:to_mask]
            perm = torch.randperm(len(obs_indices))  # for pose
            mask_pose = perm[:to_mask]
            mask = {"patches": mask_patches, "action": mask_action, "pose": mask_pose}
            label = 0.0
            if torch.rand(1).item() > 0.5:
                label = 1.0
                # shuffle the input
                patch_length = patch.size(0)
                noise = torch.rand(patch_length)  # noise in [0, 1]
                ids_shuffle = torch.argsort(
                    noise
                )  # ascend: small is keep, large is remove

                patch = torch.gather(
                    patch,
                    dim=0,
                    index=ids_shuffle.unsqueeze_(-1)
                    .unsqueeze(-1)
                    .repeat(1, patch.size(-2), patch.size(-1)),
                )
                ### shuffle cmd_vel
                cmd_vel_length = cmd_vel.size(0)
                noise = torch.rand(cmd_vel_length)  # noise in [0, 1]
                ids_shuffle = torch.argsort(
                    noise
                )  # ascend: small is keep, large is remove
                cmd_vel = torch.gather(
                    cmd_vel,
                    dim=0,
                    index=ids_shuffle.unsqueeze(-1).repeat(1, cmd_vel.size(-1)),
                )
                ### shuffle pose
                pose_length = pose.size(0)
                # ic(patch.shape)
                noise = torch.rand(pose_length)  # noise in [0, 1]
                ids_shuffle = torch.argsort(
                    noise
                )  # ascend: small is keep, large is remove
                pose = torch.gather(
                    pose,
                    dim=0,
                    index=ids_shuffle.unsqueeze(-1).repeat(1, pose.size(-1)),
                )

            return (patch, next_patch, cmd_vel, pose, label, mask)
        return (patch, next_patch, cmd_vel, pose)


class VertiEncoderDownStream(TvertiDatasetBase):
    def __init__(
        self,
        root: str,
        stats: str,
        train: bool = True,
        block_size: int = 6,
        data_frequency: int = 10,
        model_frequency: int = 3,
        obs_len: int = 2,
        task: str = "fkd",
        f_size: int = 7,
        height_diff: int = 0.5,
        pred_len: int = 0,
        resize: int = 14,
    ):
        self.obs_len = obs_len * data_frequency
        super().__init__(
            root=root,
            stats=stats,
            train=train,
            f_size=f_size,
            height_diff=height_diff,
            pred_len=(pred_len * model_frequency) + 1,
            resize=resize,
            obs_len=self.obs_len,
        )
        self.data_frequency = data_frequency
        self.model_frequency = model_frequency
        self.pred_len = pred_len
        self.block_size = block_size
        self.task = task

    def __len__(self):
        return (
            len(self.metadata["pose"])
            - ((self.pred_len * self.model_frequency) + 1)
            - self.obs_len
        )

    def __getitem__(self, idx: int):
        # patch input to the model
        if self.train:
            obs_indices = torch.sort(
                torch.randperm(self.obs_len)[: self.block_size]
            ).values.tolist()  # with observation randomization
        else:
            obs_indices = (
                torch.arange(
                    self.model_frequency, self.obs_len + 1, self.model_frequency
                )
            ).tolist()  # no observation randomization
        pred_indices = (
            torch.arange(
                self.model_frequency,
                (self.pred_len * self.model_frequency) + 1,
                self.model_frequency,
            )
            + self.obs_len
        ).tolist()
        obs_pred_indices = obs_indices + pred_indices

        patch = torch.stack(
            [
                (
                    self.transform(
                        read_patch(
                            self.root.parents[0] / self.metadata["footprint"][idx][i],
                            self.metadata["pose"][idx][i][2],
                            self.height_diff,
                        )
                    )
                )
                for i in obs_indices
            ],
            dim=0,
        )
        cmd_vel = to_tensor(
            [self.metadata["cmd_vel"][idx][i] for i in obs_pred_indices]
        )
        verti_pose = to_tensor(
            [self.metadata["pose_diff"][idx][i] for i in obs_pred_indices]
        )

        if self.task == "fkd":
            next_pose = to_tensor(
                [self.metadata["pose_diff"][idx][i] for i in pred_indices]
            )

            return patch, cmd_vel, verti_pose, next_pose  # pose)
        elif self.task == "bc":
            next_cmd = to_tensor(
                [self.metadata["cmd_vel"][idx][i] for i in pred_indices]
            )
            return (
                patch,
                cmd_vel,
                verti_pose,
                next_cmd,
            )  # next cmd_vel
        elif self.task == "ikd":
            next_cmd = to_tensor(
                [self.metadata["cmd_vel"][idx][i] for i in pred_indices]
            )

            return patch, cmd_vel, verti_pose, next_cmd  # next_pose, pose)
        elif self.task == "reconstruction":
            # TODO: needs rewrite
            next_patch = self.transform(
                read_patch(
                    self.root.parents[0]
                    / self.metadata["footprint"][idx][pred_indices[-1]],
                    self.metadata["pose"][idx][pred_indices[-1]][2],
                    self.height_diff,
                )
            )

            next_cmd_vel = to_tensor(self.metadata["cmd_vel"][idx][pred_indices[-1]])

            return patch, cmd_vel, verti_pose, next_patch  # , next_cmd_vel)


class VertiDecoderDataset(TvertiDatasetBase):
    def __init__(
        self,
        root: str,
        stats: str,
        train: bool = True,
        block_size: int = 6,
        frequency: int = 10,
        obs_len: int = 2,
        f_size: int = 7,
        height_diff: int = 0.5,
        pred_len: int = 1,
        resize: int = 14,
        skip_frames: int = 1,
    ):
        self.obs_len = obs_len * frequency
        super().__init__(
            root=root,
            stats=stats,
            train=train,
            f_size=f_size,
            height_diff=height_diff,
            pred_len=(skip_frames * pred_len) + 1,
            resize=resize,
            obs_len=self.obs_len,
        )
        self.pred_len = pred_len
        self.frequency = frequency
        self.block_size = block_size
        self.skip_frames = skip_frames

    def __len__(self):
        return (
            len(self.metadata["pose"])
            - ((self.skip_frames * self.pred_len) + 1)
            - self.obs_len
        )

    def __getitem__(self, idx: int):
        # patch input to the model
        if self.train:
            obs_indices = (
                torch.arange(self.skip_frames, self.obs_len + 1, self.skip_frames)
            ).tolist()[
                -self.block_size :
            ]  # no observation randomization
            pred_indices = list(map(lambda x: x + self.skip_frames, obs_indices))
        else:  # during evaluation
            obs_indices = (
                torch.arange(self.skip_frames, self.obs_len + 1, self.skip_frames)
            ).tolist()[
                -self.block_size :
            ]  # no observation randomization
            pred_indices = [
                obs_indices[-1] + (i * self.skip_frames)
                for i in range(1, self.pred_len + 1)
            ]

        patch = torch.stack(
            [
                (
                    self.transform(
                        read_patch(
                            self.root.parents[0] / self.metadata["footprint"][idx][i],
                            self.metadata["pose"][idx][i][2],
                            self.height_diff,
                        )
                    )
                )
                for i in obs_indices
            ],
            dim=0,
        )
        future_patch = torch.stack(
            [
                (
                    self.transform(
                        read_patch(
                            self.root.parents[0] / self.metadata["footprint"][idx][i],
                            self.metadata["pose"][idx][i][2],
                            self.height_diff,
                        )
                    )
                )
                for i in pred_indices
            ],
            dim=0,
        )
        cmd_vel = to_tensor([self.metadata["cmd_vel"][idx][i] for i in obs_indices])
        future_cmd_vel = to_tensor(
            [self.metadata["cmd_vel"][idx][i] for i in pred_indices]
        )
        verti_pose = to_tensor(
            [self.metadata["pose_diff"][idx][i] for i in obs_indices]
        )
        future_verti_pose = to_tensor(
            [self.metadata["pose_diff"][idx][i] for i in pred_indices]
        )
        return (
            patch,
            cmd_vel,
            verti_pose,
            future_patch,
            future_cmd_vel,
            future_verti_pose,
        )


class VertiFormerDataset(TvertiDatasetBase):
    def __init__(
        self,
        root: str,
        stats: str,
        train: bool = True,
        block_size: int = 6,
        data_frequency: int = 10,
        model_frequency: int = 3,
        obs_len: int = 2,
        f_size: int = 7,
        height_diff: int = 0.5,
        pred_len: int = 0,
        resize: int = 14,
    ):
        self.obs_len = obs_len * data_frequency
        super().__init__(
            root=root,
            stats=stats,
            train=train,
            f_size=f_size,
            height_diff=height_diff,
            pred_len=(pred_len * model_frequency) + 1,
            resize=resize,
            obs_len=self.obs_len,
        )
        self.data_frequency = data_frequency
        self.model_frequency = model_frequency
        self.pred_len = pred_len
        self.block_size = block_size

    def __len__(self):
        return (
            len(self.metadata["pose"])
            - ((self.pred_len * self.model_frequency) + 1)
            - self.obs_len
        )

    def __getitem__(self, idx: int):
        # patch input to the model
        if self.train:
            obs_indices = (
                torch.arange(
                    self.model_frequency, self.obs_len + 1, self.model_frequency
                )
            ).tolist()[
                : self.block_size
            ]  # no observation randomization
        else:  # during evaluation
            obs_indices = (
                torch.arange(
                    self.model_frequency, self.obs_len + 1, self.model_frequency
                )
            ).tolist()[
                : self.block_size
            ]  # no observation randomization
        pred_indices = (
            torch.arange(
                self.model_frequency,
                (self.pred_len * self.model_frequency) + 1,
                self.model_frequency,
            )
            + self.obs_len
        ).tolist()
        obs_pred_indices = obs_indices + pred_indices

        patch = torch.stack(
            [
                (
                    self.transform(
                        read_patch(
                            self.root.parents[0] / self.metadata["footprint"][idx][i],
                            self.metadata["pose"][idx][i][2],
                            self.height_diff,
                        )
                    )
                )
                for i in obs_pred_indices  # range(self.block_size + self.pred_len)
            ],
            dim=0,
        )
        cmd_vel = to_tensor(
            [self.metadata["cmd_vel"][idx][i] for i in obs_pred_indices]
        )
        verti_pose = to_tensor(
            [self.metadata["pose_diff"][idx][i] for i in obs_pred_indices]
        )
        return patch, cmd_vel, verti_pose


class VanillaDownStream(TvertiDatasetBase):
    def __init__(
        self,
        root: str,
        stats: str,
        train: bool = True,
        task: str = "pose",
        f_size: int = 7,
        height_diff: int = 0.5,
        pred_len: int = 1,
    ):
        super().__init__(
            root=root,
            stats=stats,
            train=train,
            f_size=f_size,
            height_diff=height_diff,
            pred_len=pred_len,
        )
        self.task = task

    def __len__(self):
        return len(self.metadata["pose"]) - 1

    def __getitem__(self, idx: int):
        pred_indices = (torch.arange(3, (self.pred_len**2) + 1, 3) + 1).tolist()
        # patch input to the model
        patch = read_patch(
            self.root.parents[0] / self.metadata["footprint"][idx][0],
            self.metadata["pose"][idx][0][2],
            self.height_diff,
        )
        cmd_vel = to_tensor(self.metadata["cmd_vel"][idx][0])
        if self.task == "fkd":
            next_pose = to_tensor(
                [self.metadata["pose_diff"][idx][i] for i in pred_indices]
            )
            return patch, cmd_vel, next_pose
        elif self.task == "bc":
            next_cmd = to_tensor(
                [self.metadata["cmd_vel"][idx][i] for i in pred_indices]
            )
            return (
                patch,
                cmd_vel,
                next_cmd,
            )  # next cmd_vel
        elif self.task == "ikd":
            next_cmd = to_tensor(
                [self.metadata["cmd_vel"][idx][i] for i in pred_indices]
            )
            next_pose = to_tensor(self.metadata["pose_diff"][idx][1])
            return patch, cmd_vel, (next_cmd, next_pose)


if __name__ == "__main__":
    pass
