"""Cartesian impedance control for Franka Emika Panda robot."""

import os

import numpy as np
import pinocchio as pin
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from franesis.utils.utils import EvalRecorder


class HFICController:
    """Hybrid Force Impedance Controller."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, freq: int = 100):
        """Initialize the controller.

        Args:
            obs: The initial observation of the environment's state
            info: Additional environment information from the reset.
            freq: Control frequency in Hz.
        """
        self.freq = freq
        self.dt = 1.0 / freq
        self.steps = 0
        # Initialize evaluation recorder
        self.eval_recorder = EvalRecorder()

        # 1. stiffness and damping gains
        self.kp = np.array([1000.0] * 3 + [24.0] * 3)  # position stiffness
        self.kd = np.array([120.0] * 3 + [10.0] * 3)  # velocity damping
        # nullspace control parameters
        self.kp_null = 100.0
        self.kd_null = 20.0
        self.last_F_ext = obs["F_ext"].copy()
        # force control parameters
        self.kp_force = 1.2
        self.kd_force = 0.003

        # 2. import robot model with Pinocchio for kinematics/dynamics computations
        self.mjcf_path = info.get("mjcf_path", "franesis/envs/franka_emika_panda/panda_cylinder.xml")
        self.mjcf_path = os.path.abspath(self.mjcf_path)

        # Build pinocchio model from MJCF (fixed base by default)
        self.model = pin.buildModelFromMJCF(self.mjcf_path)
        self.data = self.model.createData()

        # 3. desired setpoint (can be updated online in compute_control)
        self.q_home = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
        trajectory_center = np.array([0.3, 0.0, 0.3])
        # Figure-8 trajectory
        num_loops = 2
        self.trajectory_time = 3.0 * num_loops
        n_steps = int(np.ceil(self.trajectory_time * self.freq))
        t = np.linspace(0, 2 * np.pi * num_loops, n_steps)
        radius = 0.2  # Radius for the circles
        t_dot = 2 * np.pi * num_loops / self.trajectory_time
        x = radius / 2 * np.sin(2 * t)
        y = radius * np.sin(t)
        z = np.zeros_like(t)
        self.trajectory = np.array([x, y, z]).T + trajectory_center
        d_x = radius * np.cos(2 * t) * t_dot
        d_y = radius * np.cos(t) * t_dot
        d_z = np.zeros_like(t)
        self.trajectory_vel = np.array([d_x, d_y, d_z]).T
        dd_x = -2 * radius * np.sin(2 * t) * t_dot**2
        dd_y = -radius * np.sin(t) * t_dot**2
        dd_z = np.zeros_like(t)
        self.trajectory_acc = np.array([dd_x, dd_y, dd_z]).T
        self.quat_des = np.array([0.0, 0.0, -1.0, 0.0])
        self.force_des = np.array([0.0, 0.0, -40.0, 0.0, 0.0, 0.0])

    def compute_control(self, obs: dict[str, NDArray[np.floating]], info: dict | None = None) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

        Args:
            obs: The current observation of the environment.
            info: Optional additional information as a dictionary.

        Returns:
            The desired joint torques as a numpy array.
        """
        # 1. prepare data
        q = obs["q"]
        dq = obs["dq"]
        pos = obs["ee_pos"]
        quat = obs["ee_quat"]
        J = info["ee_jacobian"]
        J_force = info.get("ee_jacobian_force", J[..., 2:3, :])  # (1, n)
        J_motion = info.get("ee_jacobian_motion", np.concatenate([J[..., :2, :], J[..., 3:, :]], axis=-2))  # (5, n)
        U, S, Vt = np.linalg.svd(J)
        J_null = Vt[-1:, :]  # (1, n)
        J_null = J_null / np.linalg.norm(J_null)

        dx = J @ dq
        idx = min(self.steps, self.trajectory.shape[0] - 1)
        pos_des = self.trajectory[idx]
        vel_des = self.trajectory_vel[idx]
        tau_ctrl = np.zeros_like(q)

        # M(q)
        M = pin.crba(self.model, self.data, q)
        M = 0.5 * (M + M.T)  # force symmetry

        # # C(q, dq)
        # C = pin.computeCoriolisMatrix(self.model, self.data, q, dq)
        # Cdq = C @ dq
        # # g(q)
        # g = pin.computeGeneralizedGravity(self.model, self.data, q)

        # 2. compansate for nonlinear effects
        # nonlinear effects = C*dq + g
        nle = pin.nonLinearEffects(self.model, self.data, q, dq)
        tau_ctrl += nle

        # 3. nullspace control
        I_n = np.eye(self.model.nq)
        I_m = np.eye(J.shape[0])
        J_task = J  # (m,n)
        M_inv = np.linalg.solve(M, I_n)
        Lambda = np.linalg.inv(J_task @ M_inv @ J_task.T + 1e-4 * I_m)  # (m,m)
        J_sharp = M_inv @ J_task.T @ Lambda  # (n,m)
        N = I_n - J_sharp @ J_task  # (n,n)
        tau_null = -self.kp_null * (q - self.q_home) - self.kd_null * dq
        tau_ctrl += N.T @ tau_null

        # 4. cartesian impedance control
        R_act = R.from_quat(quat).as_matrix()
        R_des = R.from_quat(self.quat_des).as_matrix()
        R_delta = R_act.T @ R_des  # compute SO(3) error
        eR = R.from_matrix(R_delta).as_rotvec()
        eR = R_act.T @ eR  # convert to world frame

        x_tilde = np.concatenate([pos - pos_des, eR])
        dx_tilde = dx - np.concatenate([vel_des, np.zeros(3)])
        F_imp = -self.kp * x_tilde - self.kd * dx_tilde
        tau_ctrl += J.T @ F_imp

        # 5. force control
        F_ext = obs["F_ext"]
        F_des = self.force_des
        dF_ext = (F_ext - self.last_F_ext) / self.dt
        self.last_F_ext = F_ext.copy()
        F_force = F_des + self.kp_force * (F_ext + F_des) - self.kd_force * dF_ext
        tau_ctrl += J.T @ F_force

        return tau_ctrl

    def step_callback(
        self, action: NDArray[np.floating], obs: dict[str, NDArray[np.floating]], reward: float, done: bool, info: dict
    ):
        """Record data and increment step counter."""
        # Record data with batch dimension (1, dim)
        idx = min(self.steps, self.trajectory.shape[0] - 1)
        position = obs["ee_pos"].copy()
        goal = self.trajectory[idx].copy()
        rpy = R.from_quat(obs["ee_quat"]).as_euler("xyz")

        action = info.get("actions", np.zeros((4,)))
        self.eval_recorder.record_step(
            action=action[None, :],
            position=position[None, :],
            goal=goal[None, :],
            rpy=rpy[None, :],
            force=obs["F_ext"][None, :3],
            goal_force=-self.force_des[None, :3],
        )

        self.steps += 1

    def episode_callback(self, exp_name: str = "default_hifc"):
        """Plot data."""
        self.steps = 0
        self.eval_recorder.plot_eval(save_path=f"{exp_name}_plot.png")
