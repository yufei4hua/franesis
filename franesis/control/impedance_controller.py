"""Cartesian impedance control for Franka Emika Panda robot."""

import os

import numpy as np
import pinocchio as pin
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from franesis.utils.utils import EvalRecorder


class CartesianImpedanceController:
    """Example of a controller using the collective thrust and attitude interface."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, freq: int = 100):
        """Initialize the attitude controller.

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
        self.kp = np.array([800.0] * 3 + [32.0] * 3)  # position stiffness
        self.kd = np.array([80.0] * 3 + [16.0] * 3)  # velocity damping
        # 2. import robot model with Pinocchio for kinematics/dynamics computations
        self.mjcf_path = info.get("mjcf_path", "franesis/envs/franka_emika_panda/panda_cylinder.xml")
        self.mjcf_path = os.path.abspath(self.mjcf_path)

        # Build pinocchio model from MJCF (fixed base by default)
        self.model = pin.buildModelFromMJCF(self.mjcf_path)
        self.data = self.model.createData()

        # Basic dimension checks (Panda fixed-base usually nq=nv=7)
        q0 = obs["q"]
        dq0 = obs["dq"]
        assert q0.shape[0] == self.model.nq, (q0.shape, self.model.nq)
        assert dq0.shape[0] == self.model.nv, (dq0.shape, self.model.nv)

        # 3. desired setpoint (can be updated online in compute_control)
        # Figure-8 trajectory
        num_loops = 2
        self.trajectory_time = 5 * num_loops
        n_steps = int(np.ceil(self.trajectory_time * self.freq))
        t = np.linspace(0, 2 * np.pi * num_loops, n_steps)
        radius = 0.2  # Radius for the circles
        t_dot = 2 * np.pi * num_loops / self.trajectory_time
        x = radius / 2 * np.sin(2 * t) + 0.3
        y = radius * np.sin(t) + 0.0
        z = np.zeros_like(t) + 0.48
        self.trajectory = np.array([x, y, z]).T
        d_x = radius * np.cos(2 * t) * t_dot
        d_y = radius * np.cos(t) * t_dot
        d_z = np.zeros_like(t)
        self.trajectory_vel = np.array([d_x, d_y, d_z]).T
        dd_x = -2 * radius * np.sin(2 * t) * t_dot**2
        dd_y = -radius * np.sin(t) * t_dot**2
        dd_z = np.zeros_like(t)
        self.trajectory_acc = np.array([dd_x, dd_y, dd_z]).T
        self.pos_des = np.array([0.3, 0.0, 0.3])
        self.quat_des = R.from_euler("xyz", [0, 180, 0], degrees=True).as_quat()

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

        # 3. cartesian impedance control law
        R_act = R.from_quat(quat).as_matrix()
        R_des = R.from_quat(self.quat_des).as_matrix()
        R_delta = R_des.T @ R_act  # compute SO(3) error
        eR = R.from_matrix(R_delta).as_rotvec()
        eR = R_act.T @ eR  # convert to world frame

        x_tilde = np.concatenate([pos - pos_des, eR])
        dx_tilde = dx - np.concatenate([vel_des, np.zeros(3)])
        F_imp = -self.kp * x_tilde - self.kd * dx_tilde
        tau_ctrl += J.T @ F_imp

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
            goal_force=np.zeros((1, 3)),
        )

        self.steps += 1

    def episode_callback(self, exp_name: str = "default_imp"):
        """Plot data."""
        self.steps = 0
        self.eval_recorder.plot_eval(save_path=f"{exp_name}_plot.png")
