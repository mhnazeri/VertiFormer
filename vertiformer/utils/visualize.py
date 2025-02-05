"""Utility functions related to vision"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from transforms3d.euler import euler2mat


def _motionvec_to_rotation(motionvec: np.ndarray) -> np.ndarray:
    """Create a rotation matrix from a motion vector

    Args:
        motionvec (np.ndarray): motion vector in (x, y, z, r, p, y)

    Returns:
        np.ndarray: rotation matrix
    """
    if isinstance(motionvec, torch.Tensor):
        motionvec = motionvec.numpy()

    translation = motionvec[:3]
    rot = R.from_rotvec(motionvec[3:])
    rot_matrix = np.eye(4)
    rot_matrix[:3, :3] = rot.as_matrix()
    rot_matrix[:3, 3] = translation
    return rot_matrix


def _motion_to_pose(motions):
    """
    Transforms a sequence of pose deltas into actual poses.

    Args:
        motions: List of pose deltas.

    Returns:
        List of poses (4x4 transformation matrices).
    """
    # initial pose is always 0, 0, 0 (robot has not moved yet)
    initial_pose = _motionvec_to_rotation(np.array([0, 0, 0, 0, 0, 0]))
    poses = [initial_pose]
    current_pose = initial_pose
    for delta in motions:
        current_pose = current_pose @ _motionvec_to_rotation(
            delta
        )  # Matrix multiplication
        poses.append(current_pose)

    return poses


def visualize_trajectory(
    poses: torch.Tensor, gt_poses: torch.Tensor = None, orientation: bool = False
):
    """
    Visualizes a sequence of robot poses in 3D with color coding based on time.

    Args:
        poses: A list of tuples, each representing a robot predicted motion in the format (x, y, z, roll, pitch, yaw).
        gt_poses: A list of tuples, each representing a robot actual motion in the format (x, y, z, roll, pitch, yaw).
        orientation: whether to visualize the orientation or not
    Example:
        poses = [
                    (0, 0, 0, 0, 0, 0),
                    (1, 1, 0, 0, 0, np.pi/4),
                    (2, 2, 1, 0, np.pi/4, 0),
                    (3, 3, 2, np.pi/4, 0, 0)
                ]
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    if isinstance(poses, torch.Tensor):
        poses = poses.cpu().detach().squeeze().tolist()
    if (gt_poses is not None) and isinstance(gt_poses, torch.Tensor):
        gt_poses = gt_poses.cpu().detach().squeeze().tolist()

    # transform motion to pose
    poses = _motion_to_pose(poses)

    # Extract x, y, z coordinates and timestamps
    x = [pose[0, -1] for pose in poses]
    y = [pose[1, -1] for pose in poses]
    z = [pose[2, -1] for pose in poses]

    if gt_poses is not None:
        gt_poses = _motion_to_pose(gt_poses)
        x_gt = [pose[0, -1] for pose in gt_poses]
        y_gt = [pose[1, -1] for pose in gt_poses]
        z_gt = [pose[2, -1] for pose in gt_poses]

    ax.plot(
        x,
        y,
        z,
        c="red",
        marker="^",
        markersize=4,
        linestyle="dashed",
        linewidth=2,
        label="Predicted Trajectory",
    )

    if gt_poses is not None:
        ax.plot(
            x_gt,
            y_gt,
            z_gt,
            c="green",
            marker="v",
            markersize=4,
            linestyle="solid",
            linewidth=2,
            label="GroundTruth Trajectory",
        )

    # Optionally, visualize robot orientations at some key poses
    if orientation:
        for i in range(1, len(poses), 1):  # skip the initial pose
            pose = poses[i]
            u, v, w = pose[:, 0], pose[:, 1], pose[:, 2]  # Get axis vectors
            scale = 0.1  # Adjust scale as needed
            ax.quiver(
                pose[0, -1],
                pose[1, -1],
                pose[2, -1],
                u[0] * scale,
                u[1] * scale,
                u[2] * scale,
                color="blue",
            )
            ax.quiver(
                pose[0, -1],
                pose[1, -1],
                pose[2, -1],
                v[0] * scale,
                v[1] * scale,
                v[2] * scale,
                color="red",
            )
            ax.quiver(
                pose[0, -1],
                pose[1, -1],
                pose[2, -1],
                w[0] * scale,
                w[1] * scale,
                w[2] * scale,
                color="green",
            )

            if gt_poses is not None:
                gt_pose = gt_poses[i]
                # You might need to adjust the scale of the orientation visualization
                u_gt, v_gt, w_gt = (
                    gt_pose[:, 0],
                    gt_pose[:, 1],
                    gt_pose[:, 2],
                )  # Get axis vectors
                ax.quiver(
                    gt_pose[0, -1],
                    gt_pose[1, -1],
                    gt_pose[2, -1],
                    u_gt[0] * scale,
                    u_gt[1] * scale,
                    u_gt[2] * scale,
                    color="blue",
                )
                ax.quiver(
                    gt_pose[0, -1],
                    gt_pose[1, -1],
                    gt_pose[2, -1],
                    v_gt[0] * scale,
                    v_gt[1] * scale,
                    v_gt[2] * scale,
                    color="red",
                )
                ax.quiver(
                    gt_pose[0, -1],
                    gt_pose[1, -1],
                    gt_pose[2, -1],
                    w_gt[0] * scale,
                    w_gt[1] * scale,
                    w_gt[2] * scale,
                    color="green",
                )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.axis(xmin=-0.5, xmax=2, ymin=-1.5, ymax=1.5, zmin=-1.5, zmax=1.5)
    ax.legend()
    return fig
