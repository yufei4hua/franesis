"""Franka Core Env for low-level control and state access."""

import genesis as gs
import torch


class FrankaCore:
    def __init__(self, num_envs: int, scene: gs.Scene, device: str = "cpu") -> None:
        self._device = device
        self._scene = scene
        self._num_envs = num_envs

        material = gs.materials.Rigid()
        morph = gs.morphs.MJCF(
            file="franesis/envs/franka_emika_panda/panda_cylinder.xml", pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0)
        )
        self.robot: gs.Entity = scene.add_entity(material=material, morph=morph)

        self._default_arm_q = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
        self._default_gripper_q = [0.04, 0.04]

        self._init_indices()

    def _init_indices(self):
        self._arm_dof_dim = min(7, self.robot.n_dofs)
        self._gripper_dim = max(0, self.robot.n_dofs - self._arm_dof_dim)

        # Match default qpos dimensions to the current robot model (with/without gripper).
        self._default_arm_q = self._default_arm_q[: self._arm_dof_dim]
        self._default_gripper_q = self._default_gripper_q[: self._gripper_dim]

        self._arm_dof_idx = torch.arange(self._arm_dof_dim, device=self._device)
        self._fingers_dof = torch.arange(self._arm_dof_dim, self._arm_dof_dim + self._gripper_dim, device=self._device)

        self._ee_link = self.robot.get_link("attachment")

    def _set_pd_gains(self):
        kp_arm = [4500, 4500, 3500, 3500, 2000, 2000, 2000]
        kv_arm = [450, 450, 350, 350, 200, 200, 200]
        fr_min_arm = [-87, -87, -87, -87, -12, -12, -12]
        fr_max_arm = [87, 87, 87, 87, 12, 12, 12]

        kp = kp_arm + ([100] * self._gripper_dim)
        kv = kv_arm + ([10] * self._gripper_dim)
        fr_min = fr_min_arm + ([-100] * self._gripper_dim)
        fr_max = fr_max_arm + ([100] * self._gripper_dim)

        self.robot.set_dofs_kp(torch.tensor(kp, dtype=gs.tc_float, device=self._device))
        self.robot.set_dofs_kv(torch.tensor(kv, dtype=gs.tc_float, device=self._device))
        self.robot.set_dofs_force_range(
            torch.tensor(fr_min, dtype=gs.tc_float, device=self._device),
            torch.tensor(fr_max, dtype=gs.tc_float, device=self._device),
        )

    def _reset_home(self, mask: torch.Tensor | None = None):
        if mask is None:
            mask = torch.arange(self._num_envs, device=self._device)

        q = torch.tensor(self._default_arm_q + self._default_gripper_q, dtype=gs.tc_float, device=self._device).repeat(
            len(mask), 1
        )
        self.robot.set_qpos(q, envs_idx=mask)

    def _apply_force(self, tau: torch.Tensor):
        full_tau = torch.zeros((self._num_envs, self.robot.n_dofs), device=self._device, dtype=gs.tc_float)
        full_tau[:, : self._arm_dof_dim] = tau
        full_tau[:, self._fingers_dof] = 0.0
        self.robot.control_dofs_force(force=full_tau)

    def _get_q(self) -> torch.Tensor:
        return self.robot.get_dofs_position(dofs_idx_local=self._arm_dof_idx)

    def _get_dq(self) -> torch.Tensor:
        return self.robot.get_dofs_velocity(dofs_idx_local=self._arm_dof_idx)

    def _get_ee_pose(self) -> torch.Tensor:
        return self._ee_link.get_pos(), self._ee_link.get_quat()

    def _get_jacobian_ee(self) -> torch.Tensor:
        return self.robot.get_jacobian(link=self._ee_link)
