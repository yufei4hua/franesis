"""Plot evaluation CSV data for different controllers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import fire
import matplotlib.pyplot as plt
import numpy as np


def _load_csv(csv_path: Path) -> np.ndarray:
    return np.genfromtxt(csv_path, delimiter=",", names=True, dtype=float)


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values**2)))


def _extract_controller_name(csv_path: Path, environment: str) -> str:
    stem = csv_path.stem  # e.g., "box_HFIC_plot"
    prefix = f"{environment}_"
    suffix = "_plot"
    if stem.startswith(prefix) and stem.endswith(suffix):
        return stem[len(prefix) : -len(suffix)]
    return stem


def plot_environment(environment: str = "box") -> Tuple[Path, Path]:
    root_dir = Path(__file__).parents[1]
    save_dir = root_dir / "saves"
    csv_files = sorted(save_dir.glob(f"{environment}_*_plot.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found for environment '{environment}' in {save_dir}")

    controller_data: Dict[str, np.ndarray] = {}
    for csv_path in csv_files:
        controller = _extract_controller_name(csv_path, environment)
        controller_data[controller] = _load_csv(csv_path)

    # Define controller order
    controller_order = ["imp", "mfc", "hfic", "pdimfc"]
    sorted_controllers = [c for c in controller_order if c in controller_data]

    # environment name
    match environment:
        case "box":
            env_name = "Planar Surface"
        case "surface":
            env_name = "Curved Surface"

    # Plot Z Force comparison
    fig_force, ax_force = plt.subplots(figsize=(6, 4))
    colors = plt.cm.tab10.colors
    for idx, controller in enumerate(sorted_controllers):
        data = controller_data[controller]
        force_z = data["force_2"]
        goal_force_z = data["goal_force_2"]
        err = force_z - goal_force_z
        rmse_z = _rmse(err)
        ax_force.plot(force_z, color=colors[idx % len(colors)], label=f"{controller.upper()} (RMSE: {rmse_z:.3f} N)")

    # Plot goal force once (assume same length as data)
    last_data = list(controller_data.values())[-1]
    ax_force.plot(last_data["goal_force_2"], color="black", linestyle="--", label="Goal")
    ax_force.set_title(f"{env_name} - Normal Force")
    ax_force.set_xlabel("Time Step")
    ax_force.set_ylabel("Force (N)")
    ax_force.grid(True)
    ax_force.legend()

    z_force_path = save_dir / f"{environment}_force.png"
    fig_force.tight_layout()
    fig_force.savefig(z_force_path)

    # Plot Trajectory Plane (XY)
    fig_traj, ax_traj = plt.subplots(figsize=(6, 6))
    for idx, controller in enumerate(sorted_controllers):
        data = controller_data[controller]
        pos_x = data["pos_0"]
        pos_y = data["pos_1"]
        goal_x = data["goal_0"]
        goal_y = data["goal_1"]
        err_xy = np.sqrt((pos_x - goal_x) ** 2 + (pos_y - goal_y) ** 2)
        rmse_xy = _rmse(err_xy)
        ax_traj.plot(
            pos_x,
            pos_y,
            color=colors[idx % len(colors)],
            linewidth=3,
            label=f"{controller.upper()} (RMSE: {rmse_xy * 1000:.3f} mm)",
        )

    ax_traj.plot(last_data["goal_0"], last_data["goal_1"], color="black", linestyle="--", label="Goal")
    ax_traj.set_title(f"{env_name} - Trajectory Plane (XY)")
    ax_traj.set_xlabel("X Position (m)")
    ax_traj.set_ylabel("Y Position (m)")
    ax_traj.grid(True)
    ax_traj.legend()
    ax_traj.axis("equal")
    ax_traj.set_xlim(-0.15, 0.3)

    traj_path = save_dir / f"{environment}_motion.png"
    fig_traj.tight_layout()
    fig_traj.savefig(traj_path)

    return z_force_path, traj_path


if __name__ == "__main__":
    fire.Fire(plot_environment, serialize=lambda _: None)
