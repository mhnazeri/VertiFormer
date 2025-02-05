from pathlib import Path
import pickle
from collections import defaultdict

import numpy as np
import torch
from torchvision.transforms import v2

from helpers import to_tensor, read_patch, read_map


def calculate_stats(root: str, f_size: int = 7, height_diff: int = 0.5):

    root = Path(root)
    with open(root, "rb") as f:
        metadata = pickle.load(f)

    data = defaultdict(list)

    transform = v2.Compose(
        [
            v2.Resize(size=(16, 16), antialias=True),
            v2.ToDtype(torch.float32, scale=False),
        ]
    )
    num_samples = 0
    for i, bag_name in enumerate(metadata["bag_name"]):
        num_samples += len(metadata["data"][i]["cmd_vel"])
        data["cmd_vel"].extend(to_tensor(metadata["data"][i]["cmd_vel"]))
        # self.metadata["elevation_map"].extend(metadata["data"][i]["elevation_map"])
        data["footprint"].extend(metadata["data"][i]["footprint"])
        data["pose"].extend(to_tensor(metadata["data"][i]["pose"]))
        data["motor_speed"].extend(to_tensor(metadata["data"][i]["motor_speed"]))
        data["dt"].extend(to_tensor(metadata["data"][i]["dt"]))
        data["pose_diff"].extend(to_tensor(metadata["data"][i]["pose_diff"]))
        data["time"].extend(to_tensor(metadata["data"][i]["time"]))

    data["cmd_vel"] = torch.stack(data["cmd_vel"], dim=0)
    data["footprint"] = torch.stack(
        [
            transform(read_patch(root.parents[0] / footprint, pose[2]))
            for footprint, pose in zip(data["footprint"], data["pose"])
        ],
        dim=0,
    )
    data["pose"] = torch.stack(data["pose"], dim=0)
    data["motor_speed"] = torch.stack(data["motor_speed"], dim=0)
    data["dt"] = torch.stack(data["dt"], dim=0)
    data["pose_diff"] = torch.stack(data["pose_diff"], dim=0)
    data["time"] = torch.stack(data["time"], dim=0)

    stats = {"name": root.parent.name}
    for key, value in data.items():
        # print(f"{key = }")
        if key in ["footprint", "elevation_map"]:
            stats[key + "_mean"] = torch.mean(value)
            stats[key + "_var"] = torch.var(value)
            stats[key + "_std"] = torch.std(value)
            stats[key + "_max"] = torch.max(value)
            stats[key + "_min"] = torch.min(value)
        else:
            stats[key + "_mean"] = torch.mean(value, dim=0)
            stats[key + "_var"] = torch.var(value, dim=0)
            stats[key + "_std"] = torch.std(value, dim=0)
            stats[key + "_max"] = torch.max(value, dim=0).values
            stats[key + "_min"] = torch.min(value, dim=0).values

    for key, value in stats.items():
        print(f"{key}: {value}")

    with open(root.parents[0] / "stats.pkl", "wb") as f:
        pickle.dump(stats, f)


if __name__ == "__main__":
    calculate_stats("vertiformer/data/data_train.pickle")
