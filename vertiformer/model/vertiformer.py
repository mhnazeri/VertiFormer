"""VertiFormer model definition"""

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
from vertiformer.model.vertiencoder import VertiEncoder


class VertiFormer(nn.Module):
    def __init__(self, cfg: DictConfig, encoder: nn.Module = None):
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder  # VertiEncoder
        self.block_size = cfg.transformer.block_size * 2

        self.emb_dim = cfg.transformer_layer.d_model

        # modality encoder and decoders
        self.action_encoder = nn.Linear(2, self.emb_dim // 2, bias=False)
        self.pose_encoder = nn.Linear(6, self.emb_dim // 2, bias=False)
        self.pose_decoder = nn.Linear(self.emb_dim, 6, bias=False)
        self.action_decoder = nn.Linear(self.emb_dim, 2, bias=False)
        self.patch_encoder = nn.Linear(self.emb_dim, self.emb_dim)

        ### positional encoding
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
        self.pre_norm = nn.RMSNorm(self.emb_dim // 2)
        ### VertiDecoder ###
        transformer_decoder = nn.TransformerDecoderLayer(**cfg.transformer_layer)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=transformer_decoder, num_layers=cfg.transformer.num_layers
        )
        self.mid_norm = nn.RMSNorm(self.emb_dim)
        self.post_norm = nn.RMSNorm(self.emb_dim)
        self.patch_decoder = nn.Linear(self.emb_dim, self.emb_dim)
        self.patch_encoder.weight = self.patch_decoder.weight  # weight tying
        ### learnable masking for missing information and causal masking
        mask = torch.tril(torch.ones(self.cfg.pred_len, self.cfg.pred_len)).unsqueeze(0)
        self.register_buffer("mask", mask)
        self.input_mask_actions = nn.Parameter(
            torch.zeros(1, self.cfg.pred_len, self.emb_dim // 2)
        )
        self.input_mask_poses = nn.Parameter(
            torch.zeros(1, self.cfg.pred_len, self.emb_dim // 2)
        )

        # initialize weights
        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith("out_proj.weight"):
                nn.init.normal_(
                    param,
                    mean=0.0,
                    std=0.02 / math.sqrt(2 * cfg.transformer.num_layers),
                )

    def _init_weights(self, module):
        nn.init.xavier_normal_(self.input_mask_actions)
        nn.init.xavier_normal_(self.input_mask_poses)
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0, 0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        patches: torch.Tensor,
        actions: torch.Tensor,
        poses: torch.Tensor,
        future_actions: torch.Tensor | None = None,
        future_poses: torch.Tensor | None = None,
        task: str = "all",
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            patches: patches of terrain in (B, T, H, W)
            actions: low-level actions in (B, T, 2)
            poses: pose of the robot on terrain in (B, T, 6)
            future_actions: a list of planned future actions in (B, T, 2)
            future_poses: a list of planned future poses in (B, T, 6)
        Returns:
            pred_poses: predicted future pose of the robot on terrain in (B, T, 6)
            pred_actions: predicted future low-level actions in (B, T, 2)
        """
        b, t, _ = poses.shape
        # get the context tokens from VertiEncoder
        embeddings_BTC = self.patch_encoder(
            self.encoder.encode(patches, actions, poses)
        )
        # check which modality is missing and replace with the learnable mask
        if future_poses is not None:  # IKD  case
            future_poses = self.pre_norm(
                self.pose_encoder(future_poses)
            )  # (B, T, 6) -> (B, T, C)
        else:
            future_poses = self.input_mask_poses.expand(b, -1, -1)
        if future_actions is not None:  # FKD case
            future_actions = self.pre_norm(
                self.action_encoder(future_actions)
            )  # (B, T, 2) -> (B, T, C)
        else:
            future_actions = self.input_mask_actions.expand(b, -1, -1)

        # f_s unified representation
        query = torch.cat([future_actions, future_poses], dim=-1)
        query = self.patch_encoder(query)

        # Query VertiDecoder with context tokens for cross-attention and causal-masking
        tokens = self.transformer_decoder(
            query,
            embeddings_BTC,
            tgt_mask=self.mask,
            memory_mask=self.mask,
            tgt_is_causal=True,
            memory_is_causal=True,
        )
        tokens = self.mid_norm(tokens)
        tokens = self.post_norm(self.patch_decoder(tokens))
        pred_poses = self.pose_decoder(tokens)
        pred_actions = self.action_decoder(tokens)
        return pred_poses, pred_actions, 0


class MMVertiFormer(nn.Module):
    def __init__(self, cfg: DictConfig, encoder: nn.Module = None):
        """VertiFormer with patch head prediction"""
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder
        self.block_size = cfg.transformer.block_size
        self.patch_size = cfg.patch_size
        self.emb_dim = cfg.transformer_layer.d_model

        # modality encoders and decoders
        self.action_encoder = nn.Linear(2, self.emb_dim // 2, bias=False)
        self.pose_encoder = nn.Linear(6, self.emb_dim // 2, bias=False)
        self.pose_decoder = nn.Linear(self.emb_dim, 6, bias=False)
        self.action_decoder = nn.Linear(self.emb_dim, 2, bias=False)
        self.patch_encoder = nn.Linear(self.emb_dim, self.emb_dim)
        self.img_decoder = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim * 2),
            nn.GELU(),
            nn.Linear(self.emb_dim * 2, self.emb_dim * 2),
            nn.GELU(),
            nn.Linear(self.emb_dim * 2, cfg.transformer_layer.d_model * 2),
            nn.GELU(),
            nn.Linear(cfg.transformer_layer.d_model * 2, cfg.patch_size**2),
        )

        ### positional encoding
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
        self.pre_norm = nn.RMSNorm(self.emb_dim // 2)
        ### VertiDecoder
        transformer_decoder = nn.TransformerDecoderLayer(**cfg.transformer_layer)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=transformer_decoder, num_layers=cfg.transformer.num_layers
        )
        self.mid_norm = nn.RMSNorm(self.emb_dim)
        self.post_norm = nn.RMSNorm(self.emb_dim)
        self.patch_decoder = nn.Linear(self.emb_dim, self.emb_dim)
        ### learnable masking for missing information and causal masking
        mask = torch.tril(torch.ones(self.cfg.pred_len, self.cfg.pred_len)).unsqueeze(0)
        self.input_mask_actions = nn.Parameter(
            torch.zeros(1, self.cfg.pred_len, self.emb_dim // 2)
        )
        self.input_mask_poses = nn.Parameter(
            torch.zeros(1, self.cfg.pred_len, self.emb_dim // 2)
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
        nn.init.xavier_uniform_(self.input_mask_actions)
        nn.init.xavier_uniform_(self.input_mask_poses)
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, 0, 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        patches: torch.Tensor,
        actions: torch.Tensor,
        poses: torch.Tensor,
        future_actions: torch.Tensor | None = None,
        future_poses: torch.Tensor | None = None,
        task: str = "all",
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            patches: patches of terrain in (B, T, H, W)
            actions: low-level actions in (B, T, 2)
            poses: pose of the robot on terrain in (B, T, 6)
            future_actions: a list of planned future actions in (B, T, 2)
            future_poses: a list of planned future poses in (B, T, 6)
            task: type of data we want to return, options: ['all', 'fkd', 'ikd', 'patch']
        Returns:
            future_poses: future pose of the robot on terrain in (B, T, 6)
            future_actions: future low-level actions in (B, T, 2)
        """
        b, t, _ = poses.shape
        # get the context tokens from VertiEncoder
        embeddings_BTC = self.patch_encoder(
            self.encoder.encode(patches, actions, poses)
        )

        # mask the missing modality
        if future_poses is not None:  # IKD  case
            future_poses = self.pre_norm(
                self.pose_encoder(future_poses)
            )  # (B, T, 6) -> (B, T, C)
        else:
            future_poses = self.input_mask_poses.expand(b, -1, -1)
        if future_actions is not None:  # FKD case
            future_actions = self.pre_norm(
                self.action_encoder(future_actions)
            )  # (B, T, 2) -> (B, T, C)
        else:
            future_actions = self.input_mask_actions.expand(b, -1, -1)

        # f_s unified representation mapping
        query = torch.cat([future_actions, future_poses], dim=-1)

        query = self.patch_encoder(query)
        tokens = self.transformer_decoder(
            query,
            embeddings_BTC,
            tgt_mask=self.mask,
            memory_mask=self.mask,
            tgt_is_causal=True,
            memory_is_causal=True,
        )
        tokens = self.mid_norm(tokens)
        # map back to the same unified representation
        tokens = self.post_norm(self.patch_decoder(tokens))
        # decode to the modality
        pred_poses = self.pose_decoder(tokens)
        pred_actions = self.action_decoder(tokens)
        pred_images = self.img_decoder(tokens).view(
            b, -1, self.patch_size, self.patch_size
        )
        if task == "fkd":
            return pred_poses, None, None
        if task == "ikd":
            return None, pred_actions, None
        if task == "patch":
            return None, None, pred_images
        if task == "all":
            return pred_poses, pred_actions, pred_images


def load_model(cfg):
    """Initializes vertiDecoder model"""
    vertiencoder = VertiEncoder(cfg.model.vertiencoder)
    cfg.model.encoder_weights = False
    if cfg.model.vertiencoder_weight:
        cfg.model.encoder_weights = True
        encoder_weights = torch.load(cfg.model.vertiencoder_weight, weights_only=False)[
            "model"
        ]
        unwanted_prefix = "_orig_mod."
        for k, v in list(encoder_weights.items()):
            if k.startswith(unwanted_prefix):
                encoder_weights[k[len(unwanted_prefix) :]] = encoder_weights.pop(k)
        vertiencoder.load_state_dict(encoder_weights)
        # freeze the model
        for param in vertiencoder.parameters():
            param.requires_grad_(False)
        vertiencoder.eval()

    if cfg.train_params.model_type == "uni":
        model = VertiFormer(cfg.model, encoder=vertiencoder)
    elif cfg.train_params.model_type == "multi":
        model = MMVertiFormer(cfg.model, encoder=vertiencoder)
    else:
        raise TypeError(f"Provided model type {cfg.model_type} is not supported.")

    optimizer = optim.AdamW(model.parameters(), **cfg.adamw)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.train_params.epochs
    )
    return model, optimizer, scheduler
