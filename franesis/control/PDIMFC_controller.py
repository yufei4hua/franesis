"""Cartesian impedance control for Franka Emika Panda robot."""

import os

import numpy as np
import pinocchio as pin
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from franesis.utils.utils import EvalRecorder


class PDIMFCController:
    """Partial Decoupled Impedance Motion Force Controller"""

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
        # motion control parameters
        self.kp = np.array([800.0] * 3 + [80.0] * 3)  # position stiffness
        self.kd = np.array([80.0] * 3 + [8.0] * 3)  # velocity damping
        # force control parameters
        self.kp_force = np.array([0.0, 0.0, 3.0] + [0.0] * 3)
        self.kd_force = np.array([0.0, 0.0, 0.002] + [0.0] * 3)
        # nullspace control parameters
        self.kp_null = 100.0
        self.kd_null = 20.0
        self.last_F_ext = obs["F_ext"].copy()

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
        # J = info.get("ee_jacobian_task", info["ee_jacobian"])
        J_force = info["ee_jacobian_force"]
        J_motion = info["ee_jacobian_motion"]
        J_null = info["ee_jacobian_null"]
        J_mf = np.vstack([J_motion, J_force])  # (6, 7)
        motion_idx = [0, 1, 3, 4, 5]
        idx = min(self.steps, self.trajectory.shape[0] - 1)
        pos_des = self.trajectory[idx]
        vel_des = self.trajectory_vel[idx]
        acc_des = self.trajectory_acc[idx]
        tau_ctrl = np.zeros_like(q)

        # M(q)
        # M = pin.crba(self.model, self.data, q)
        M_inv = pin.computeMinverse(self.model, self.data, q)
        # C(q, dq)
        C = pin.computeCoriolisMatrix(self.model, self.data, q, dq)
        pin.computeJointJacobiansTimeVariation(self.model, self.data, q, dq)
        dJ_full = pin.getFrameJacobianTimeVariation(
            self.model, self.data, self.model.getFrameId("tool_tip"), pin.LOCAL_WORLD_ALIGNED
        )
        dJ_mf = np.vstack([dJ_full[motion_idx, :], dJ_full[2:3, :]])  # (6,7)
        dJdq = dJ_full @ dq
        dM = C + C.T
        dM_inv = -M_inv @ dM @ M_inv
        dLambda_inv = (
            dJ_mf @ M_inv @ J_mf.T
            + J_mf @ dM_inv @ J_mf.T
            + J_mf @ M_inv @ dJ_mf.T
        )  # (6,6)

        # 2. compansate for nonlinear effects
        # nonlinear effects = C*dq + g
        nle = pin.nonLinearEffects(self.model, self.data, q, dq)
        tau_ctrl += nle

        # 3. nullspace control
        F_null = J_null @ (-self.kp_null * (q - self.q_home) - self.kd_null * dq)
        tau_ctrl += J_null.T @ F_null

        # 4. choose inertia
        Lambda_inv = J_mf @ M_inv @ J_mf.T
        Lambda = np.linalg.inv(Lambda_inv + 1e-6 * np.eye(6))
        Lambda_des = Lambda.copy()
        A = Lambda[:5, :5]
        B = Lambda[:5, 5:6]
        D = Lambda[5:6, 5:6]
        Lambda_des[5:6, :5] = 0.0 # partial decoupling
        Lambda_des[5:6, 5:6] = D - B.T @ np.linalg.inv(A + 1e-6 * np.eye(5)) @ B
        Lambda_des_inv = np.linalg.inv(Lambda_des)
        Lambda_motion_des = Lambda_des[:5, :5]
        Lambda_couple_des = Lambda_des[:5, 5:6]
        Lambda_dot = -Lambda @ dLambda_inv @ Lambda  # (6,6)
        C_bar = 0.5 * Lambda_dot  # (6,6)
        C_m = C_bar[:5, :5]       # (5,5)
        C_f = C_bar[5:6, 5:6]     # (1,1)

        # 4. cartesian impedance control (u_m)
        R_act = R.from_quat(quat).as_matrix()
        R_des = R.from_quat(self.quat_des).as_matrix()
        R_delta = R_des.T @ R_act  # compute SO(3) error
        eR = R.from_matrix(R_delta).as_rotvec()
        eR = R_act.T @ eR  # convert to world frame

        x = np.concatenate([pos[:2], eR, pos[2:3]])  # (6,)
        x_des = np.concatenate([pos_des[:2], np.zeros_like(eR), pos_des[2:3]])  # (6,)
        x_tilde = x - x_des
        dx = J_mf @ dq  # (6,)
        dx_des = np.concatenate([vel_des[:2], np.zeros(3), vel_des[2:3]])
        dx_tilde = dx - dx_des
        dd_x_des = np.concatenate([acc_des[:2], np.zeros(3), acc_des[2:3]])
        x_m_tilde = x_tilde[:5]
        dx_m_tilde = dx_tilde[:5]
        dd_x_m_des = dd_x_des[:5]
        dd_x_f_des = dd_x_des[5:6]

        u_m = Lambda_motion_des @ dd_x_m_des + Lambda_couple_des @ dd_x_f_des \
            - self.kp[motion_idx] * x_m_tilde - self.kd[motion_idx] * dx_m_tilde \
            + C_m @ dx_m_tilde

        # 5. force control (u_f)
        F_ext = info.get("ee_task_F_ext", obs["F_ext"])
        F_mf_ext = F_ext[[0, 1, 3, 4, 5, 2]]  # reorder to match J_mf
        F_f_ext = F_ext[2:3]
        F_des = self.force_des[2:3]
        dF_ext = (F_ext - self.last_F_ext) / self.dt
        dF_f_ext = dF_ext[2:3]
        self.last_F_ext = F_ext.copy()
        
        u_f = F_des + self.kp_force[2:3] * (F_f_ext + F_des) - self.kd_force[2:3] * dF_f_ext \
            + C_f.squeeze() * dx[5:6]
        
        # 6. shape inertia
        u = np.concatenate([u_m, u_f])  # (6,)
        F = Lambda @ (Lambda_des_inv @ u - dJdq) + (Lambda @ Lambda_des_inv - np.eye(6)) @ F_mf_ext
        
        tau_ctrl += J_mf.T @ F

        return tau_ctrl

    def step_callback(
        self, action: NDArray[np.floating], obs: dict[str, NDArray[np.floating]], reward: float, done: bool, info: dict
    ):
        """Record data and increment step counter."""
        # Record data with batch dimension (1, dim)
        position = info.get("ee_task_pos", obs["ee_pos"])
        quat = info.get("ee_task_quat", obs["ee_quat"])
        quat = obs["ee_quat"]
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
