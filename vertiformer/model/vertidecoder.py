"""VertiDecoder model definition"""

from typing import Tuple
import math

import torch
from torch import nn
from torch import optim
from omegaconf import DictConfig

from vertiformer.model.positional_encoding import (
    LearnablePositionalEncoding,
    SinusoidalPositionalEncoding
)


class VertiDecoder(nn.Module):
    def __init__(self, cfg: DictConfig, encoder: nn.Module = None):
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg.transformer.block_size
        self.patch_size = cfg.patch_size
        self.emb_dim = cfg.transformer_layer.d_model

        self.action_encoder = nn.Linear(2, self.emb_dim // 4, bias=False)
        self.pose_encoder = nn.Linear(6, self.emb_dim // 4, bias=False)
        self.pose_decoder = nn.Linear(self.emb_dim, 6, bias=False)
        self.action_decoder = nn.Linear(self.emb_dim, 2, bias=False)
        self.patch_encoder = nn.Linear(self.emb_dim, self.emb_dim)
        self.img_encoder = nn.Linear(cfg.patch_size**2, self.emb_dim // 2)
        self.img_decoder = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim * 2),
            nn.GELU(),
            nn.Linear(self.emb_dim * 2, self.emb_dim * 2),
            nn.GELU(),
            nn.Linear(self.emb_dim * 2, self.emb_dim * 2),
            nn.GELU(),
            nn.Linear(self.emb_dim * 2, cfg.patch_size**2),
        )

        if cfg.pos_type == "learnable":
            self.pos = LearnablePositionalEncoding(
                max_len=self.block_size,
                d_model=self.emb_dim,
                dropout=cfg.pos_encoding.dropout,
            )
        elif cfg.pos_type == "sinusoidal":
            self.pos = SinusoidalPositionalEncoding(
                max_len=self.block_size,
                d_model=self.emb_dim,
                dropout=cfg.pos_encoding.dropout,
            )
        else:
            raise TypeError(
                f"Provided positional encoding type {cfg.pos_type} is not supported."
            )
        self.pre_norm = nn.RMSNorm(self.emb_dim)
        transformer_decoder = nn.TransformerEncoderLayer(**cfg.transformer_layer)
        self.transformer_decoder = nn.TransformerEncoder(
            encoder_layer=transformer_decoder,
            num_layers=cfg.transformer.num_layers,
            enable_nested_tensor=False,
        )
        self.mid_norm = nn.RMSNorm(self.emb_dim)
        self.post_norm = nn.RMSNorm(self.emb_dim)
        self.patch_decoder = nn.Linear(self.emb_dim, self.emb_dim)
        self.patch_decoder.weight = self.patch_encoder.weight  # weight tying
        mask = torch.tril(torch.ones(self.block_size, self.block_size)).view(
            1, self.block_size, self.block_size
        )
        self.register_buffer("mask", mask)

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
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, 0, 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(
        self, patches: torch.Tensor, actions: torch.Tensor, poses: torch.Tensor
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            patches: patches of terrain in (B, T, H, W)
            actions: low-level actions in (B, T, 2)
            poses: pose of the robot on terrain in (B, T, 6)
        Returns:
            future_poses: future pose of the robot on terrain in (B, T, 6)
            future_actions: future low-level actions in (B, T, 2)
            future_patches: future patches of terrain in (B, T, p_size, p_size)
        """
        b, t, _ = poses.shape
        actions = self.action_encoder(actions)
        poses = self.pose_encoder(poses)
        patches = patches.flatten(
            start_dim=2
        )  # (B, T, H, W) -> (B, T, patch_size ** 2)
        patches = self.img_encoder(patches)
        # concat tokens together
        tokens = torch.cat([patches, actions, poses], dim=-1)

        tokens = self.pre_norm(tokens)
        tokens = self.patch_encoder(tokens)
        # since we care about the token that is not part of the sequence during validation, we can ignore masking
        mask = (
            self.mask.expand(b * self.cfg.transformer_layer.nhead, -1, -1)
            if self.training
            else None
        )
        tokens = self.transformer_decoder(tokens, mask=mask)
        tokens = self.mid_norm(tokens)
        tokens = self.post_norm(self.patch_decoder(tokens))
        pred_poses = self.pose_decoder(tokens)
        pred_actions = self.action_decoder(tokens)
        pred_images = self.img_decoder(tokens).view(
            b, t, self.patch_size, self.patch_size
        )
        if self.training:
            return pred_poses, pred_actions, pred_images
        else:
            return (
                pred_poses[:, [-1], :],
                pred_actions[:, [-1], :],
                pred_images[:, [-1], :],
            )

    @torch.inference_mode()
    def generate(
        self,
        patches: torch.Tensor,
        actions: torch.Tensor,
        poses: torch.Tensor,
        steps: int = 3,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for i in range(steps):
            # query the model
            pred_poses, pred_actions, pred_images = self(
                patches[:, i:, :], actions[:, i:, :], poses[:, i:, :]
            )
            patches = torch.cat([patches, pred_images], dim=1)
            actions = torch.cat([actions, pred_actions], dim=1)
            poses = torch.cat([poses, pred_poses], dim=1)

        return poses[:, -steps:, :], actions[:, -steps:, :], patches[:, -steps:, :]


def load_model(cfg):
    """Initializes vertiDecoder model"""
    model = VertiDecoder(cfg.model)
    optimizer = optim.AdamW(model.parameters(), **cfg.adamw)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.train_params.epochs
    )
    return model, optimizer, scheduler
