"""Cartesian impedance control for Franka Emika Panda robot."""

import os

import numpy as np
import pinocchio as pin
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R


class CartesianImpedanceController:
    """Example of a controller using the collective thrust and attitude interface."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, freq: int = 100):
        """Initialize the attitude controller.

        Args:
            obs: The initial observation of the environment's state
            info: Additional environment information from the reset.
        """
        self.freq = freq
        self.dt = 1.0 / freq

        # 1. stiffness and damping gains
        self.kp = np.array([200.0, 200.0, 200.0, 30.0, 30.0, 30.0])  # position stiffness
        self.kd = np.array([20.0, 20.0, 20.0, 10.0, 10.0, 10.0])  # velocity damping

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
        self.pos_des = np.array([0.3, 0.1, -0.05])
        self.quat_des = np.array([0.0, 0.0, -1.0, 0.0])

    def compute_control(self, obs: dict[str, NDArray[np.floating]], info: dict | None = None) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The collective thrust and orientation [t_des, r_des, p_des, y_des] as a numpy array.
        """
        # 1. prepare data
        q = obs["q"]
        dq = obs["dq"]
        pos = obs["ee_pos"]
        quat = obs["ee_quat"]
        J = info["ee_jacobian"]
        dx = J @ dq
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
        # R_delta = (R.from_quat(self.quat_des).inv() * R.from_quat(quat)).as_matrix()
        # R_act = R.from_quat(quat).as_matrix()
        # R_des = R.from_quat(self.quat_des).as_matrix()
        # R_delta = R_des.T @ R_act
        # eRM = R_delta - R_delta.T
        # eR = np.stack((eRM[..., 2, 1], eRM[..., 0, 2], eRM[..., 1, 0]), axis=-1)
        R_des = R.from_quat(self.quat_des).as_matrix()
        R_act = R.from_quat(quat).as_matrix()

        # error in desired
        R_err = R_des.T @ R_act
        eR_des = R.from_matrix(R_err).as_rotvec()

        # convert to world to match dx/world-J
        eR = -R_des @ eR_des

        print("des quat:", self.quat_des)
        print("curr quat:", quat)
        print("eR:", eR)
        x_tilde = np.concatenate([pos - self.pos_des, eR])
        F_imp = -self.kp * x_tilde - self.kd * dx
        tau_ctrl += J.T @ F_imp

        return tau_ctrl
