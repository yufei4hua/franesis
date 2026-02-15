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
        self.kp = np.array([800.0] * 3 + [80.0] * 3)  # position stiffness
        self.kd = np.array([80.0] * 3 + [8.0] * 3)  # velocity damping
        # nullspace control parameters
        self.kp_null = 100.0
        self.kd_null = 20.0
        self.last_F_ext = obs["F_ext"].copy()
        # force control parameters
        self.kp_force = np.array([0.0, 0.0, 3.0] + [0.0] * 3)
        self.kd_force = np.array([0.0, 0.0, 0.002] + [0.0] * 3)

        # 2. import robot model with Pinocchio for kinematics/dynamics computations
        self.mjcf_path = info.get("mjcf_path", "franesis/envs/franka_emika_panda/panda_cylinder.xml")
        self.mjcf_path = os.path.abspath(self.mjcf_path)

        # Build pinocchio model from MJCF (fixed base by default)
        self.model = pin.buildModelFromMJCF(self.mjcf_path)
        self.data = self.model.createData()

        # 3. desired setpoint
        self.q_home = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
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
        self.trajectory = np.array([x, y, z]).T
        d_x = radius * np.cos(2 * t) * t_dot
        d_y = radius * np.cos(t) * t_dot
        d_z = np.zeros_like(t)
        self.trajectory_vel = np.array([d_x, d_y, d_z]).T
        dd_x = -2 * radius * np.sin(2 * t) * t_dot**2
        dd_y = -radius * np.sin(t) * t_dot**2
        dd_z = np.zeros_like(t)
        self.trajectory_acc = np.array([dd_x, dd_y, dd_z]).T
        self.quat_des = R.from_euler("yxz", [180, 0, 0], degrees=True).as_quat()
        self.force_des = np.array([0.0, 0.0, -10.0, 0.0, 0.0, 0.0])

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
        pos = info.get("ee_task_pos", obs["ee_pos"])
        quat = info.get("ee_task_quat", obs["ee_quat"])
        J = info.get("ee_jacobian_task", info["ee_jacobian"])
        J_force = info["ee_jacobian_force"]
        J_motion = info["ee_jacobian_motion"]
        # J_task = info["ee_jacobian_task"]
        J_null = info["ee_jacobian_null"]
        # J_bar = np.vstack([J_force, J_motion, J_null])  # (7, 7)
        # J_bar_inv = np.linalg.inv(J_bar)
        motion_idx = [0, 1, 3, 4, 5]
        idx = min(self.steps, self.trajectory.shape[0] - 1)
        pos_des = self.trajectory[idx]
        vel_des = self.trajectory_vel[idx]
        acc_des = self.trajectory_acc[idx]
        tau_ctrl = np.zeros_like(q)

        # M(q)
        M = pin.crba(self.model, self.data, q)
        M_inv = np.linalg.inv(M)
        M_x = np.linalg.inv(J_motion @ M_inv @ J_motion.T)
        Lambda_force = np.linalg.inv(J_force @ M_inv @ J_force.T + 1e-6 * np.eye(1))
        # C(q, dq)
        C = pin.computeCoriolisMatrix(self.model, self.data, q, dq)
        # C_inv = np.linalg.inv(C + 1e-6 * np.eye(7))
        pin.computeJointJacobiansTimeVariation(self.model, self.data, q, dq)
        dJ_full = pin.getFrameJacobianTimeVariation(
            self.model, self.data, self.model.getFrameId("tool_tip"), pin.LOCAL_WORLD_ALIGNED
        )
        # dJ_motion = dJ_full[motion_idx, :]  # (5, 7)
        dJ_force = dJ_full[2:3, :]  # (1, 7)
        # C_x = np.linalg.inv(dJ_motion @ M_inv @ J_motion.T + 1e-6 * np.eye(5)) + np.linalg.inv(
        #     J_motion @ C_inv @ J_motion.T + 1e-6 * np.eye(5)
        # )  # (5, 5)
        # g = pin.computeGeneralizedGravity(self.model, self.data, q)

        # 2. compansate for nonlinear effects
        # nonlinear effects = C*dq + g
        nle = pin.nonLinearEffects(self.model, self.data, q, dq)
        tau_ctrl += nle

        # 3. nullspace control
        F_null = J_null @ (-self.kp_null * (q - self.q_home) - self.kd_null * dq)
        tau_ctrl += J_null.T @ F_null

        # 4. cartesian impedance control
        R_act = R.from_quat(quat).as_matrix()
        R_des = R.from_quat(self.quat_des).as_matrix()
        R_delta = R_des.T @ R_act  # compute SO(3) error
        eR = R.from_matrix(R_delta).as_rotvec()
        eR = R_act.T @ eR  # convert to world frame

        x = np.concatenate([pos[:2], eR])  # (5,)
        x_des = np.concatenate([pos_des[:2], np.zeros_like(eR)])
        x_tilde = x - x_des
        dx = J_motion @ dq  # (5,)
        d_x_des = np.concatenate([vel_des[:2], np.zeros(3)])
        dx_tilde = dx - d_x_des
        dd_x_des = np.concatenate([acc_des[:2], np.zeros(3)])

        F_imp = M_x @ dd_x_des - self.kp[motion_idx] * x_tilde - self.kd[motion_idx] * dx_tilde
        tau_ctrl += J_motion.T @ F_imp

        # 5. force control
        F_ext = info.get("ee_task_F_ext", obs["F_ext"])
        F_motion_ext = F_ext[motion_idx]
        F_null_ext = np.zeros_like(F_null)  # how to compute?
        F_des = self.force_des
        dF_ext = (F_ext - self.last_F_ext) / self.dt
        self.last_F_ext = F_ext.copy()
        F_force_FF = (
            F_des
            - Lambda_force @ J_force @ M_inv @ (J_motion.T @ F_imp + J_null.T @ F_null)
            + Lambda_force @ J_force @ M_inv @ (J_motion.T @ F_motion_ext + J_null.T @ F_null_ext)
            + Lambda_force @ (J_force @ M_inv @ C - dJ_force) @ dq
        )
        F_force_PD = self.kp_force * (F_ext + F_des) - self.kd_force * dF_ext
        tau_ctrl += J.T @ (F_force_FF + F_force_PD)

        return tau_ctrl

    def step_callback(
        self, action: NDArray[np.floating], obs: dict[str, NDArray[np.floating]], reward: float, done: bool, info: dict
    ):
        """Record data and increment step counter."""
        # Record data with batch dimension (1, dim)
        position = info.get("ee_task_pos", obs["ee_pos"])
        quat = info.get("ee_task_quat", obs["ee_quat"])
        quat = obs["ee_quat"] # plot world frame
        idx = min(self.steps, self.trajectory.shape[0] - 1)
        goal = self.trajectory[idx].copy()
        pry = R.from_quat(quat).as_euler("yxz")
        rpy = np.array([pry[1], pry[0], pry[2]])
        force = info.get("ee_task_F_ext", obs["F_ext"])

        action = info.get("actions", np.zeros((4,)))
        self.eval_recorder.record_step(
            action=action[None, :],
            position=position[None, :],
            goal=goal[None, :],
            rpy=rpy[None, :],
            force=force[None, :3],
            goal_force=-self.force_des[None, :3],
        )

        self.steps += 1

    def episode_callback(self, exp_name: str = "default_hifc"):
        """Plot data."""
        self.steps = 0
        self.eval_recorder.plot_eval(save_path=f"{exp_name}_plot.png")
