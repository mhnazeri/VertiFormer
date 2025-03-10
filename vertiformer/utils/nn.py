"""Utils for nn module"""

from typing import List, Optional
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from thop import profile, clever_format


@torch.no_grad()
def check_grad_norm(net: nn.Module):
    """Compute and return the grad norm of all parameters of the network.
    To check whether gradients flowing in the network or not
    """
    total_norm = 0
    for p in list(filter(lambda p: p.grad is not None, net.parameters())):
        param_norm = p.grad.detach().norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1.0 / 2)
    return total_norm


def make_mlp(
    dims: list,
    act: str,
    l_act: bool = False,
    norm: str = "ln",
    dropout: float = 0.0,
    bias: bool = False,
):
    """Create a simple MLP with batch-norm and dropout

    Args:
        dims: (List) a list containing the dimensions of MLP
        act: (str) activation function to be used. Valid activations are [relu, tanh, sigmoid]
        l_act: (bool) whether to use activation after the last linear layer
        bn: (bool) use batch-norm or not. Default is True
        dropout: (float) dropout percentage
    """
    layers = []
    activation = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "elu": nn.ELU(),
        "gelu": nn.GELU(),
        "leakyrelu": nn.LeakyReLU(negative_slope=0.2),
    }[act.lower()]

    if norm:
        normalization = {
            "bn": nn.BatchNorm1d,
            "ln": nn.LayerNorm,
            "rms": nn.RMSNorm,
        }[norm.lower()]

    for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(in_dim, out_dim, bias=bias))
        if i != (len(dims) - 2):
            if norm:
                layers.append(normalization(out_dim))

            layers.append(activation)

            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))

    if l_act:
        layers.append(activation)

    return nn.Sequential(*layers)


def freeze(model: nn.Module, exclude: List = [], verbose: bool = False) -> nn.Module:
    """Freezes the layers of the model except the exclusion layer list.

    Args:
        model: (nn.Module) The model itself.
        exclude: (List) The list of layers name that you want to keep unfrozen.
        verbose: (bool) Show statistics of the model parameters.

    Returns:
        model: (nn.Module) returns the frozen model.
    """
    frozen_layers_list = []
    frozen_params_list = [len(p) for p in model.parameters() if p.requires_grad]
    if verbose:
        print(f"The model has {len([p for p in model.parameters()])} layers.")
        print(f"Before freezing the model had {sum(frozen_params_list)} parameters.")

    if len(exclude) == 0 or exclude is None:
        for name, child in model.named_parameters():
            child.requires_grad_(False)

        if verbose:
            print(
                f"The whole model with {len([p for p in model.parameters()])} layers have been frozen."
            )

        return model

    for name, child in model.named_parameters():
        if not any(layer in name for layer in exclude):
            frozen_layers_list.append(name)
            child.requires_grad_(False)

    frozen_params_list = [len(p) for p in model.parameters() if p.requires_grad]
    if verbose:
        print(f"{len(frozen_layers_list)} layers have been frozen.")
        print(f"After freezing the model has {sum(frozen_params_list)} parameters.")

    return model


@torch.no_grad()
def init_weights(
    method: str = "kaiming_normal",
    mean: float = 0.0,
    std: float = 0.5,
    low: float = 0.0,
    high: float = 1.0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
    gain: float = 1.0,
):
    """Initialize the network's weights based on the provided method

    Args:
        m: (nn.Module) module itself
        method: (str) how to initialize the weights
        mean: (float) mean of normal distribution
        std: (float) standard deviation for normal distribution
        low: (float) minimum threshold for uniform distribution
        high: (float) maximum threshold for uniform distribution
        mode: (str) either 'fan_in' (default) or 'fan_out'. Choosing 'fan_in' preserves the magnitude of the variance of the weights in the forward pass. Choosing 'fan_out' preserves the magnitudes in the backwards pass.
        nonlinearity: (str) the non-linear function (nn.functional name), recommended to use only with 'relu' or 'leaky_relu' (default).
        gain: (float) an optional scaling factor for xavier initialization
    """
    if method == "kaiming_normal":

        def init(m):
            if any(
                [
                    isinstance(m, nn.Conv1d),
                    isinstance(m, nn.Conv2d),
                    isinstance(m, nn.Conv3d),
                    isinstance(m, nn.Linear),
                ]
            ):
                nn.init.kaiming_normal_(m.weight, mode=mode, nonlinearity=nonlinearity)
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / (fan_out**0.5)
                    nn.init.normal_(m.bias, -bound, bound)

            # Kaiming_init does not work with Batchnorm
            elif any(
                [
                    isinstance(m, nn.BatchNorm1d),
                    isinstance(m, nn.BatchNorm2d),
                    isinstance(m, nn.BatchNorm3d),
                ]
            ):
                nn.init.normal_(m.weight, mean=mean, std=std)

        return init
    elif method == "kaiming_uniform_":

        def init(m):
            if any(
                [
                    isinstance(m, nn.Conv1d),
                    isinstance(m, nn.Conv2d),
                    isinstance(m, nn.Conv3d),
                    isinstance(m, nn.Linear),
                ]
            ):
                nn.init.kaiming_uniform_(m.weight, mode=mode, nonlinearity=nonlinearity)
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / (fan_out**0.5)
                    nn.init.uniform_(m.bias, -bound, bound)
            # Kaiming_init does not work with Batchnorm
            elif any(
                [
                    isinstance(m, nn.BatchNorm1d),
                    isinstance(m, nn.BatchNorm2d),
                    isinstance(m, nn.BatchNorm3d),
                ]
            ):
                nn.init.uniform_(m.weight, mean=mean, std=std)

        return init
    elif method == "normal":

        def init(m):
            if any(
                [
                    isinstance(m, nn.Conv1d),
                    isinstance(m, nn.Conv2d),
                    isinstance(m, nn.Conv3d),
                    isinstance(m, nn.BatchNorm1d),
                    isinstance(m, nn.BatchNorm2d),
                    isinstance(m, nn.BatchNorm3d),
                    isinstance(m, nn.Linear),
                ]
            ):

                nn.init.normal_(m.weight, mean=mean, std=std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        return init
    elif method == "uniform":

        def init(m):
            if any(
                [
                    isinstance(m, nn.Conv1d),
                    isinstance(m, nn.Conv2d),
                    isinstance(m, nn.Conv3d),
                    isinstance(m, nn.BatchNorm1d),
                    isinstance(m, nn.BatchNorm2d),
                    isinstance(m, nn.BatchNorm3d),
                    isinstance(m, nn.Linear),
                ]
            ):
                nn.init.uniform_(m.weight, a=low, b=high)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        return init
    elif method == "xavier_normal":

        def init(m):
            if any(
                [
                    isinstance(m, nn.Conv1d),
                    isinstance(m, nn.Conv2d),
                    isinstance(m, nn.Conv3d),
                    isinstance(m, nn.BatchNorm1d),
                    isinstance(m, nn.BatchNorm2d),
                    isinstance(m, nn.BatchNorm3d),
                    isinstance(m, nn.Linear),
                ]
            ):
                nn.init.xavier_normal_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        return init
    elif method == "xavier_uniform":

        def init(m):
            if any(
                [
                    isinstance(m, nn.Conv1d),
                    isinstance(m, nn.Conv2d),
                    isinstance(m, nn.Conv3d),
                    isinstance(m, nn.BatchNorm1d),
                    isinstance(m, nn.BatchNorm2d),
                    isinstance(m, nn.BatchNorm3d),
                    isinstance(m, nn.Linear),
                ]
            ):
                nn.init.xavier_uniform_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        return init


def op_counter(model, sample):
    model.eval()
    macs, params = profile(model, inputs=sample)
    macs, params = clever_format([macs, params], "%.3f")
    return macs, params


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.
    Code from https://github.com/Bjarten/early-stopping-pytorch
    """

    def __init__(self, patience=7, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def get_activation(activation, name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


def init_optimizer(cfg, params: list = None, optimizer_method: str = "adam"):
    """Initialize and return optimizer and learning rate scheduler.

    Args:
        cfg: (dict) configuration parameters
        model: (nn.Module) the model
        optimizer_method: (str) optimizer method
    """
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - INITIALIZING the optimizer!")

    if optimizer_method == "adamw":
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, params),
            **cfg.adamw,
        )

    elif optimizer_method == "adam":
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, params),
            **cfg.adam,
        )

    elif optimizer_method == "rmsprop":
        optimizer = optim.RMSprop(
            filter(lambda p: p.requires_grad, params),
            **cfg.rmsprop,
        )

    elif optimizer_method == "sgd":
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, params),
            **cfg.sgd,
        )

    else:
        raise ValueError(
            f"Unknown optimizer {optimizer_method}"
            + "; valid optimizers are 'adam', 'adamw', 'sgd' and 'rmsprop'."
        )

    # initialize the learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.train_params.epochs
    )

    return optimizer, scheduler
