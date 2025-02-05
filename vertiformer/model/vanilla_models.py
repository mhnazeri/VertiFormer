import torch
import torch.nn as nn
from torchvision.models import get_model

from vertiformer.utils.nn import make_mlp


class FKD(nn.Module):
    def __init__(self, cfg, pred_len):
        super().__init__()
        self.pred_len = pred_len
        self.encoder = get_model("resnet18", weights=None).train()
        self.encoder.conv1 = nn.Conv2d(
            1,
            self.encoder.conv1.out_channels,
            kernel_size=self.encoder.conv1.kernel_size,
            stride=self.encoder.conv1.stride,
            padding=self.encoder.conv1.padding,
            bias=self.encoder.conv1.bias,
        )
        cfg.dims.insert(0, 512)
        cfg.dims.append(6 * pred_len)  # append the task output to the end
        self.encoder.fc = make_mlp(**cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        return self.encoder(x).view(B, self.pred_len, -1)


class BehaviorCloning(nn.Module):
    def __init__(self, cfg, pred_len):
        super().__init__()
        self.pred_len = pred_len
        self.encoder = get_model("resnet18", weights=None).train()
        self.encoder.conv1 = nn.Conv2d(
            1,
            self.encoder.conv1.out_channels,
            kernel_size=self.encoder.conv1.kernel_size,
            stride=self.encoder.conv1.stride,
            padding=self.encoder.conv1.padding,
            bias=self.encoder.conv1.bias,
        )
        cfg.dims.insert(0, 512)
        cfg.dims.append(2 * pred_len)  # append the task output to the end
        self.encoder.fc = make_mlp(**cfg)

    def forward(self, x: torch.Tensor, g_pose: torch.Tensor = None) -> torch.Tensor:
        B = x.shape[0]
        return self.encoder(x).view(B, self.pred_len, -1)


class IKD(nn.Module):
    def __init__(self, cfg, pred_len):
        super().__init__()
        self.pred_len = pred_len
        cfg.pose.dims.insert(0, 6)
        self.goal_pose = make_mlp(**cfg.pose)
        self.encoder = get_model("resnet18", weights=None).train()
        self.encoder.conv1 = nn.Conv2d(
            1,
            self.encoder.conv1.out_channels,
            kernel_size=self.encoder.conv1.kernel_size,
            stride=self.encoder.conv1.stride,
            padding=self.encoder.conv1.padding,
            bias=self.encoder.conv1.bias,
        )
        cfg.fc.dims.insert(0, 512)
        self.encoder.fc = make_mlp(**cfg.fc)
        self.fc = nn.Linear(cfg.fc.dims[-1] + cfg.pose.dims[-1], 2 * pred_len)

    def forward(self, x: torch.Tensor, g_pose: torch.Tensor = None) -> torch.Tensor:
        B = x.shape[0]
        x = self.encoder(x)
        g_pose = self.goal_pose(g_pose)
        pose = torch.cat([x, g_pose], dim=-1)
        return self.fc(torch.relu(pose)).view(B, self.pred_len, -1)


class DeployedModel(nn.Module):
    def __init__(self, dt: nn.Module):
        super().__init__()
        self.dt = dt

    @torch.inference_mode()
    def forward(self, patches: torch.Tensor, pose: torch.Tensor = None) -> torch.Tensor:
        return self.dt(patches, pose)


def make_model(cfg):
    if cfg.task == "bc":
        bc = BehaviorCloning(cfg.bc, cfg.pred_len)
        bc_weight = torch.load(cfg.bc_weight, map_location=torch.device("cpu"))["model"]
        unwanted_prefix = "_orig_mod."
        for k, v in list(bc_weight.items()):
            if k.startswith(unwanted_prefix):
                bc_weight[k[len(unwanted_prefix) :]] = bc_weight.pop(k)

        bc.load_state_dict(bc_weight)
        bc.eval()
        bc.requires_grad = False
        model = DeployedModel(bc)
    elif cfg.task == "fkd":
        fkd = FKD(cfg.fkd, cfg.pred_len)
        fkd_weight = torch.load(cfg.fkd_weight, map_location=torch.device("cpu"))[
            "model"
        ]
        unwanted_prefix = "_orig_mod."
        for k, v in list(fkd_weight.items()):
            if k.startswith(unwanted_prefix):
                fkd_weight[k[len(unwanted_prefix) :]] = fkd_weight.pop(k)
        fkd.load_state_dict(fkd_weight)
        fkd.eval()
        fkd.requires_grad = False
        model = DeployedModel(fkd)
    elif cfg.task == "ikd":
        ikd = IKD(cfg.ikd, cfg.pred_len)
        ikd_weight = torch.load(cfg.ikd_weight, map_location=torch.device("cpu"))[
            "model"
        ]
        unwanted_prefix = "_orig_mod."
        for k, v in list(ikd_weight.items()):
            if k.startswith(unwanted_prefix):
                ikd_weight[k[len(unwanted_prefix) :]] = ikd_weight.pop(k)

        ikd.load_state_dict(ikd_weight)
        ikd.eval()
        ikd.requires_grad = False
        model = DeployedModel(ikd)
    else:
        raise NotImplementedError(f"Unknown task {cfg.task}")

    if cfg.compile:
        model = torch.compile(model)
    return model
