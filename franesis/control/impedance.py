"""Cartesian impedance control for Franka Emika Panda robot."""

import os

import numpy as np
import pinocchio as pin
from numpy.typing import NDArray


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
        self.kp = np.array([5.0, 5.0, 10.0])  # position stiffness
        self.kd = np.array([2.0, 2.0, 5.0])  # velocity damping

        # 2. import robot model with Pinocchio for kinematics/dynamics computations
        self.mjcf_path = info.get("mjcf_path", "franesis/envs/franka_emika_panda/panda_cylinder.xml")
        self.mjcf_path = os.path.abspath(self.mjcf_path)

        # Build pinocchio model from MJCF (fixed base by default)
        self.model = pin.buildModelFromMJCF(self.mjcf_path)
        self.data = self.model.createData()

        # Basic dimension checks (Panda fixed-base usually nq=nv=7)
        q0 = obs["q"]
        dq0 = obs["dq"]
        print("Initial q:", q0, q0.shape)
        assert q0.shape[0] == self.model.nq, (q0.shape, self.model.nq)
        assert dq0.shape[0] == self.model.nv, (dq0.shape, self.model.nv)

    def compute_control(self, obs: dict[str, NDArray[np.floating]], info: dict | None = None) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The collective thrust and orientation [t_des, r_des, p_des, y_des] as a numpy array.
        """
        q  = np.asarray(obs["q"],  dtype=np.float64).reshape(-1)   # (7,)
        dq = np.asarray(obs["dq"], dtype=np.float64).reshape(-1)   # (7,)

        # M(q)
        M = pin.crba(self.model, self.data, q)
        M = 0.5 * (M + M.T)  # CRBA 数值上可能不严格对称，做一次对称化

        # C(q, dq) (matrix) and C*dq (vector)
        C = pin.computeCoriolisMatrix(self.model, self.data, q, dq)
        Cdq = C @ dq

        # g(q)
        g = pin.computeGeneralizedGravity(self.model, self.data, q)

        # (optional) nonlinear effects = C*dq + g
        nle = pin.nonLinearEffects(self.model, self.data, q, dq)

        print("M:", M)
        print("C:", C)
        print("g:", g)

        tau_ctrl = np.zeros(self.model.nv, dtype=np.float32)

        return tau_ctrl
