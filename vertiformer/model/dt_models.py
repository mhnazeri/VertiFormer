import math

import torch
import torch.nn as nn
from torch.nn import RMSNorm

from vertiformer.utils.nn import make_mlp
from vertiformer.model.vertiencoder import load_model


class FKD(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.map_cmd = nn.Sequential(
            nn.Linear(2, cfg.fc.dims[0]),
            nn.Tanh(),
            nn.Linear(cfg.fc.dims[0], 2 * cfg.fc.dims[0]),
            nn.Tanh(),
            nn.Linear(cfg.fc.dims[0] * 2, cfg.fc.dims[0]),
        )
        cfg.fc.dims.append(6)  # append the task output to the end
        cfg.fc.dims[0] = cfg.fc.dims[0] * 2
        self.fc = make_mlp(**cfg.fc)
        self.dropout = nn.Dropout(0.3)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, 0, 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(
        self, x: torch.Tensor, cmd: torch.Tensor = None, pose: torch.Tensor = None
    ) -> torch.Tensor:
        cmd = self.map_cmd(cmd)
        z = torch.cat([x, cmd], dim=-1)
        z = self.dropout(z)
        z = self.fc(z)
        return z


class BehaviorCloning(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        cfg.dims.append(2)  # append the task output to the end
        self.fc = make_mlp(**cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class IKD(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        cfg.pose.dims[0] = 6
        self.goal_pose = make_mlp(**cfg.pose)
        self.dropout = nn.Dropout(0.3)
        cfg.fc.dims.insert(0, cfg.fc.dims[0] + cfg.pose.dims[-1])
        cfg.fc.dims.append(2)  # append the task output to the end
        self.fc = make_mlp(**cfg.fc)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, 0, 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, z: torch.Tensor, g_pose: torch.Tensor = None) -> torch.Tensor:
        g_pose = self.goal_pose(g_pose)
        z = torch.cat([z, g_pose], dim=-1)
        z = self.dropout(z)
        return self.fc(z)


class DeployedModel(nn.Module):
    def __init__(self, encoder: nn.Module, downstream: nn.Module, task: str = "fkd"):
        super().__init__()
        self.encoder = encoder
        self.downstream = downstream
        self.task = task

    @torch.inference_mode()
    def forward(
        self,
        patches: torch.Tensor,
        actions: torch.Tensor,
        curr_pose: torch.Tensor = None,
        g_pose: torch.Tensor = None,
    ) -> torch.Tensor:
        ctx = self.encoder(patches, actions, curr_pose)
        if self.task == "fkd":
            return self.downstream(ctx, actions[:, -1], curr_pose[:, -1])
        elif self.task == "ikd":
            return self.downstream(ctx, g_pose, curr_pose[:, -1])
        else:
            return self.downstream(ctx)


def get_model(cfg):
    tverti, _ = load_model(cfg.tverti)
    tverti_weight = torch.load(cfg.tverti_weight, map_location=torch.device("cpu"))[
        "pretext_model"
    ]
    unwanted_prefix = "_orig_mod."
    for k, v in list(tverti_weight.items()):
        if k.startswith(unwanted_prefix):
            tverti_weight[k[len(unwanted_prefix) :]] = tverti_weight.pop(k)

    tverti.load_state_dict(tverti_weight)
    tverti = tverti.eval()
    tverti.requires_grad_(False)
    if cfg.task == "bc":
        cfg.bc.dims.insert(0, cfg.tverti.transformer_layer.d_model)
        bc = BehaviorCloning(cfg.bc)
        bc_weight = torch.load(cfg.bc_weight, map_location=torch.device("cpu"))[
            "dt_model"
        ]
        unwanted_prefix = "_orig_mod."
        for k, v in list(bc_weight.items()):
            if k.startswith(unwanted_prefix):
                bc_weight[k[len(unwanted_prefix) :]] = bc_weight.pop(k)

        bc.load_state_dict(bc_weight)
        bc = bc.eval()
        bc.requires_grad_(False)
        model = DeployedModel(tverti, bc, cfg.task)
    elif cfg.task == "fkd":
        cfg.fkd.fc.dims.insert(0, cfg.tverti.transformer_layer.d_model)
        fkd = FKD(cfg.fkd)
        fkd_weight = torch.load(cfg.fkd_weight, map_location=torch.device("cpu"))[
            "dt_model"
        ]
        unwanted_prefix = "_orig_mod."
        for k, v in list(fkd_weight.items()):
            if k.startswith(unwanted_prefix):
                fkd_weight[k[len(unwanted_prefix) :]] = fkd_weight.pop(k)
        fkd.load_state_dict(fkd_weight)
        fkd = fkd.eval()
        fkd.requires_grad_(False)
        model = DeployedModel(tverti, fkd, cfg.task)
    elif cfg.task == "ikd":
        cfg.ikd.dims.insert(0, cfg.tverti.transformer_layer.d_model)
        ikd = IKD(cfg.ikd)
        ikd_weight = torch.load(cfg.ikd_weight, map_location=torch.device("cpu"))[
            "dt_model"
        ]
        unwanted_prefix = "_orig_mod."
        for k, v in list(ikd_weight.items()):
            if k.startswith(unwanted_prefix):
                ikd_weight[k[len(unwanted_prefix) :]] = ikd_weight.pop(k)

        ikd.load_state_dict(ikd_weight)
        ikd = ikd.eval()
        ikd.requires_grad_(False)
        model = DeployedModel(tverti, ikd, cfg.task)
    else:
        raise NotImplementedError(f"Unknown task {cfg.task}")

    if cfg.compile:
        model = torch.compile(model)
    return model
