"""VertiEncoder model definition"""

from typing import Tuple
import math

import torch
from torch import nn
from omegaconf import DictConfig

from vertiformer.model.positional_encoding import (
    LearnablePositionalEncoding,
    SinusoidalPositionalEncoding
)


class VertiEncoder(nn.Module):
    def __init__(self, cfg: DictConfig, img_encoder: nn.Module = None):
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg.transformer.block_size
        self.ctx_len = cfg.pred_len
        self.emb_dim = cfg.transformer_layer.d_model
        ### modality encoders and decoders
        self.patch_size = cfg.patch_size
        self.img_encoder = nn.Linear(
            cfg.patch_size**2, cfg.transformer_layer.d_model // 2
        )
        self.img_decoder = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim * 2),
            nn.GELU(),
            nn.Linear(self.emb_dim * 2, self.emb_dim * 2),
            nn.GELU(),
            nn.Linear(self.emb_dim * 2, cfg.transformer_layer.d_model * 2),
            nn.GELU(),
            nn.Linear(cfg.transformer_layer.d_model * 2, cfg.patch_size**2),
        )
        self.action_encoder = nn.Linear(
            2, cfg.transformer_layer.d_model // 4, bias=False
        )
        self.pose_encoder = nn.Linear(6, cfg.transformer_layer.d_model // 4, bias=False)
        self.action_decoder = nn.Linear(self.emb_dim, 2, bias=False)
        self.pose_decoder = nn.Linear(self.emb_dim, 6, bias=False)
        ###
        if cfg.pos_type == "learnable":
            self.pos = LearnablePositionalEncoding(
                max_len=self.block_size + self.ctx_len,
                d_model=self.emb_dim,
                dropout=cfg.pos_encoding.dropout,
            )
        elif cfg.pos_type == "sinusoidal":
            self.pos = SinusoidalPositionalEncoding(
                max_len=self.block_size + self.ctx_len,
                d_model=self.emb_dim,
                dropout=cfg.pos_encoding.dropout,
            )
        else:
            raise TypeError(
                f"Provided positional encoding type {cfg.pos_type} is not supported."
            )
        self.pre_norm = nn.RMSNorm(self.emb_dim)
        self.patch_encoder = nn.Linear(self.emb_dim, self.emb_dim)
        cfg.transformer_layer.d_model = self.emb_dim
        transformer_encoder = nn.TransformerEncoderLayer(**cfg.transformer_layer)
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_encoder, cfg.transformer.num_layers, enable_nested_tensor=False
        )
        self.mid_norm = nn.RMSNorm(self.emb_dim)
        self.post_norm = nn.RMSNorm(self.emb_dim)
        self.patch_decoder = nn.Linear(self.emb_dim, self.emb_dim)

        self.ctx_token = nn.Parameter(torch.zeros(1, self.ctx_len, self.emb_dim))
        self.order_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(self.emb_dim), nn.Linear(self.emb_dim, 1)
        )
        # random mask
        self.register_buffer("mask_vw", torch.zeros(1, 1, 2))
        self.register_buffer("mask_pose", torch.zeros(1, 1, 6))
        # self.register_buffer('mask_patch', torch.randn(1, 1, cfg.patch_size ** 2))
        self.mask_patch = nn.Parameter(
            torch.zeros(1, 1, cfg.patch_size**2)
        )  # learnable mask for patch masking only
        # initialize weights
        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith("out_proj.weight"):
                torch.nn.init.normal_(
                    param,
                    mean=0.0,
                    std=0.02 / math.sqrt(2 * cfg.transformer.num_layers),
                )

    def _init_weights(self, module):
        nn.init.normal_(self.ctx_token, std=0.02)
        nn.init.normal_(self.mask_patch, std=0.02)
        nn.init.normal_(self.mask_vw, std=0.02)
        nn.init.normal_(self.mask_pose, std=0.02)
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, 0, 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        patches: torch.Tensor,
        actions: torch.Tensor,
        poses: torch.Tensor,
        mask: dict = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            patches: patches of terrain in (B, T, H, W)
            actions: low-level actions in (B, T, 2)
            poses: pose of the robot on terrain in (B, T, 6)
            mask: a list of tokens to be masked
        """
        b, t, _ = poses.shape
        patches = patches.flatten(
            start_dim=2
        )  # (B, T, H, W) -> (B, T, patch_size ** 2)
        if mask:
            actions[torch.arange(b).unsqueeze(1), mask["action"], :] = (
                self.mask_vw.expand(b, mask["action"].size(1), -1)
            )
            poses[torch.arange(b).unsqueeze(1), mask["pose"], :] = (
                self.mask_pose.expand(b, mask["pose"].size(1), -1)
            )
            patches[torch.arange(b).unsqueeze(1), mask["patches"], :] = (
                self.mask_patch.expand(b, mask["patches"].size(1), -1)
            )

        actions = self.action_encoder(actions)
        poses = self.pose_encoder(poses)
        patches = self.img_encoder(patches)
        # concat tokens together
        tokens = torch.cat([patches, actions, poses], dim=-1)

        tokens = self.pre_norm(tokens)
        tokens = self.patch_encoder(tokens)
        # prepend the ctx token
        ctx_token = self.ctx_token.expand(b, -1, -1)
        tokens = torch.cat([ctx_token, tokens], dim=1)
        tokens = self.pos(tokens)
        tokens = self.mid_norm(tokens)
        pred_tokens = self.transformer_encoder(tokens)
        pred_tokens = self.post_norm(pred_tokens)
        pred_tokens = self.patch_decoder(pred_tokens)

        if self.training:
            pred_label = self.order_head(
                pred_tokens[:, self.ctx_len :].flatten(start_dim=1)
            )  # check the order on the actual sequence
            next_patch_pred = self.img_decoder(pred_tokens[:, : self.ctx_len])
            next_patch_pred = next_patch_pred.view(
                b, self.cfg.pred_len, self.patch_size, self.patch_size
            )
            pred_patch_token = self.img_decoder(pred_tokens[:, self.ctx_len :]).view(
                b, t, self.patch_size, self.patch_size
            )
            pred_actions_token = self.action_decoder(pred_tokens[:, self.ctx_len :])
            pred_pose_token = self.pose_decoder(pred_tokens[:, self.ctx_len :])
            return (
                pred_tokens[:, : self.ctx_len],
                pred_label,
                next_patch_pred,
                pred_patch_token,
                pred_actions_token,
                pred_pose_token,
            )

        return pred_tokens[:, : self.ctx_len]

    def encode(
        self, patches: torch.Tensor, actions: torch.Tensor, poses: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            patches: patches of terrain in (B, T, H, W)
            actions: low-level actions in (B, T, 2)
            poses: pose of the robot on terrain in (B, T, 6)
        Returns:
            embedding tensor of shape (B, T, C)
        """
        b, t, _ = poses.shape
        patches = patches.flatten(start_dim=2)  # (B, T, H, W) -> (B, T, H * W)
        actions = self.action_encoder(actions)
        poses = self.pose_encoder(poses)
        patches = self.img_encoder(patches)
        # concat tokens together
        tokens = torch.cat([patches, actions, poses], dim=-1)
        # prepend the ctx token
        ctx_token = self.ctx_token.expand(b, -1, -1)
        tokens = torch.cat([ctx_token, tokens], dim=1)
        tokens = self.pre_norm(tokens)
        tokens = self.patch_encoder(tokens)
        tokens = self.pos(tokens)
        tokens = self.mid_norm(tokens)
        pred_tokens = self.transformer_encoder(tokens)
        pred_tokens = self.post_norm(pred_tokens)
        pred_tokens = self.patch_decoder(pred_tokens)

        return pred_tokens[:, : self.ctx_len]


class PatchEmbedding(nn.Module):
    def __init__(
        self, embedding_dim: int = 512, in_channels: int = 1, patch_size: int = 16
    ):
        """Patchify images"""
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        nn.init.kaiming_normal_(self.conv.weight)
        # scale down the effect of projection
        with torch.no_grad():
            self.conv.weight *= 0.1

    def forward(self, x):
        x = self.conv(x)
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(b, h * w, c)
        return x.squeeze()


def load_model(cfg):
    """Initializes the model"""
    model = VertiEncoder(cfg)
    return model
