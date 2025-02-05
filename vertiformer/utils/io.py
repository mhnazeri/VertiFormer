"""I/O utility functions"""

import os
import shutil

import torch


def save_checkpoint(state: dict, is_best: bool, save_dir: str, name: str):
    """Saves model and training parameters.

    Saves model and training parameters at checkpoint + 'epoch.pth'. If is_best==True, also saves
    checkpoint + 'best.pth'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        save_dir: (str) the location where to save the checkpoint
        name: (str) file name to be written
    """
    filepath = os.path.join(save_dir, f"{name}.tar")
    if not os.path.exists(save_dir):
        print(
            "Checkpoint Directory does not exist! Making directory {}".format(save_dir)
        )
        os.mkdir(save_dir)
    else:
        print(f"Checkpoint Directory exists! Saving {name} in {save_dir}")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(
            filepath, os.path.join(save_dir, f"{name.split('-')[0]}-best.tar")
        )


def load_checkpoint(save: str, device: str):
    """Loads model parameters (state_dict) from file_path.

    Args:
        save: (str) directory of the saved checkpoint
        device: (str) map location
    """
    if not os.path.exists(save):
        raise ("File doesn't exist {}".format(save))
    safe_globals = torch.serialization.get_unsafe_globals_in_checkpoint(save)
    torch.serialization.add_safe_globals(safe_globals)
    checkpoint = torch.load(save, map_location=device)

    return checkpoint
