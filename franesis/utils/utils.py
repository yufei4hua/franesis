"""Utility module."""

from dataclasses import dataclass

import matplotlib
import numpy as np

matplotlib.use("Agg")  # render to raster images
from pathlib import Path

import matplotlib.pyplot as plt
from numpy.typing import NDArray


@dataclass
class EvalRecorder:
    """Class to record evaluation data and plot them."""

    _record_act: list[NDArray]
    _record_pos: list[NDArray]
    _record_goal: list[NDArray]
    _record_rpy: list[NDArray]
    _record_force: list[NDArray]
    _record_goal_force: list[NDArray]

    def __init__(self):
        """Initialize the recorder."""
        self._record_act = []
        self._record_pos = []
        self._record_goal = []
        self._record_rpy = []
        self._record_force = []
        self._record_goal_force = []

    def record_step(
        self, action: NDArray, position: NDArray, goal: NDArray, rpy: NDArray, force: NDArray, goal_force: NDArray
    ):
        """Record a single step's data.

        Args:
            action: The action taken at this step.
            position: The drone's position at this step.
            goal: The current goal position at this step.
            rpy: The roll, pitch, yaw angles at this step.
            force: The external force at this step.
            goal_force: The goal force at this step.
        """
        self._record_act.append(action)
        self._record_pos.append(position)
        self._record_goal.append(goal)
        self._record_rpy.append(rpy)
        self._record_force.append(force)
        self._record_goal_force.append(goal_force)

    def plot_eval(self, save_path: str = "eval_plot.png", traj_plane: list = [0, 1]) -> plt.Figure:
        """Plot recorded traces and save to `save_path`."""
        pos = np.array(self._record_pos)
        goal = np.array(self._record_goal)
        rpy = np.array(self._record_rpy)
        force = np.array(self._record_force)
        goal_force = np.array(self._record_goal_force)

        fig, axes = plt.subplots(3, 4, figsize=(18, 12))
        axes = axes.flatten()

        # Position plots and goals
        for i, label in enumerate(["X Position", "Y Position", "Z Position"]):
            axes[i].plot(pos[:, 0, i])
            axes[i].plot(goal[:, 0, i], linestyle="--")
            axes[i].set_title(label)
            axes[i].set_xlabel("Time Step")
            axes[i].set_ylabel("Position (m)")
            axes[i].grid(True)
            axes[i].legend(["Position", "Goal"])

        # Position error
        pos_err = np.linalg.norm(pos[:, 0] - goal[:, 0], axis=1)
        axes[3].plot(pos_err)
        axes[3].set_title("Position Error")
        axes[3].set_xlabel("Time Step")
        axes[3].set_ylabel("Error (m)")
        axes[3].grid(True)

        # Force plots and goals
        for i, label in enumerate(["X Force", "Y Force", "Z Force"]):
            axes[4 + i].plot(force[:, 0, i])
            axes[4 + i].plot(goal_force[:, 0, i], linestyle="--")
            axes[4 + i].set_title(label)
            axes[4 + i].set_xlabel("Time Step")
            axes[4 + i].set_ylabel("Force (N)")
            axes[4 + i].grid(True)
            axes[4 + i].legend(["Force", "Goal"])

        # Force error
        force_err = np.linalg.norm(force[:, 0] - goal_force[:, 0], axis=1)
        axes[7].plot(force_err)
        axes[7].set_title("Force Error")
        axes[7].set_xlabel("Time Step")
        axes[7].set_ylabel("Error (N)")
        axes[7].grid(True)

        # Angles
        rpy_labels = ["Roll", "Pitch", "Yaw"]
        for i in range(3):
            axes[8 + i].plot(rpy[:, 0, i])
            axes[8 + i].set_title(f"{rpy_labels[i]} Angle")
            axes[8 + i].set_xlabel("Time Step")
            axes[8 + i].set_ylabel("Angle (rad)")
            axes[8 + i].grid(True)

        # compute RMSE for position
        rmse_pos = np.sqrt(np.mean(pos_err**2))
        # trajectory plot
        axes[11].plot(pos[:, 0, traj_plane[0]], pos[:, 0, traj_plane[1]], label="Actual")
        axes[11].plot(goal[:, 0, traj_plane[0]], goal[:, 0, traj_plane[1]], linestyle="--", linewidth=0.5, label="Goal")
        axes[11].set_title(f"Trajectory Plane (RMSE: {rmse_pos * 1000:.3f} mm)")
        axes[11].set_xlabel(f"{['X', 'Y', 'Z'][traj_plane[0]]} Position (m)")
        axes[11].set_ylabel(f"{['X', 'Y', 'Z'][traj_plane[1]]} Position (m)")
        axes[11].grid(True)
        axes[11].legend()
        axes[11].axis("equal")

        plt.tight_layout()
        plt.savefig(Path(__file__).parents[2] / "saves" / save_path)

        return fig
