"""Single Franka robot environment for contact tasks."""

import math

import genesis as gs
import numpy as np
import torch
from numpy.typing import NDArray

from franesis.envs.franka_core import FrankaCore


class FrankaBoxEnv(FrankaCore):
    def __init__(
        self,
        episode_length_s: float = 3.0,
        freq: int = 200,
        substeps: int = 2,
        render: bool = True,
        device: str = "cuda",
    ):
        # Box parameters
        self.size = (0.4, 0.6, 0.1)
        self.center = (0.3, 0.0, 0.3)

        super().__init__(num_envs=1, freq=freq, substeps=substeps, f_ext_lpf_alpha=0.2, render=render, device=device)
        self.max_episode_length = math.ceil(episode_length_s / self.ctrl_dt)
        self._default_arm_q = [0.13473345, -0.80271834, -0.13701877, -2.83875, -0.12417492, 2.0410793, 0.85577106]

    def _add_task_entities(self) -> None:
        # Rigid material with friction
        box_material = gs.materials.Rigid(friction=0.01, coup_softness=0.02, coup_restitution=0.0)

        # Fixed box: baselink fixed => will not be pushed away :contentReference[oaicite:3]{index=3}
        box_center = (self.center[0], self.center[1], self.center[2] - self.size[2] / 2)
        box_morph = gs.morphs.Box(pos=box_center, size=self.size, fixed=True, collision=True, visualization=True)

        # Optional: surface only affects rendering by default; physics is in material
        box_surface = gs.surfaces.Default()

        self.box: gs.Entity = self.scene.add_entity(
            material=box_material, morph=box_morph, surface=box_surface, visualize_contact=True
        )

    @staticmethod
    def _task_jocobians(
        center: torch.Tensor, ee_pos: torch.Tensor, ee_quat: torch.Tensor, ee_jacobian: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        x_task_pos = ee_pos - center
        x_task_quat = ee_quat
        J = ee_jacobian  # (B, 6, n)
        J_force = J[..., 2:3, :]  # (B, 1, n)
        J_motion = torch.cat([J[..., :2, :], J[..., 3:, :]], dim=-2)  # (B, 5, n)
        J_task = J  # (B, 6, n)
        _, _, Vh = torch.linalg.svd(J_task, full_matrices=True)
        J_null = Vh[..., -1:, :]  # (B, 1, n)
        J_null = J_null / torch.linalg.norm(J_null, dim=-1, keepdim=True).clamp_min(1e-12)
        return {
            "ee_jacobian": ee_jacobian,
            "ee_task_pos": x_task_pos,
            "ee_task_quat": x_task_quat,
            "ee_jacobian_task": J_task,
            "ee_jacobian_force": J_force,
            "ee_jacobian_motion": J_motion,
            "ee_jacobian_null": J_null,
        }

    def _reset(self, mask: torch.Tensor) -> None:
        if len(mask) == 0:
            return
        self.steps[mask] = 0

        # reset robot
        self._reset_home(mask)

    def obs(self) -> tuple[torch.Tensor, dict]:
        ee_pos, ee_quat = self._get_ee_pose()
        obs_dict = {
            "q": self._get_q(),
            "dq": self._get_dq(),
            "tau_ext": self._get_tau_ext(),
            "F_ext": self._get_ee_F_ext(),
            "ee_pos": ee_pos,
            "ee_quat": ee_quat,
        }
        return obs_dict

    def reward(self) -> torch.Tensor:
        return torch.zeros((self.num_envs,), device=self.device)

    def done(self) -> torch.Tensor:
        return self.steps > self.max_episode_length

    def info(self) -> dict:
        ee_pos, ee_quat = self._get_ee_pose()
        ee_jacobian = self._get_jacobian_ee()
        info_dict = self._task_jocobians(
            center=torch.tensor(self.center, device=gs.device), ee_pos=ee_pos, ee_quat=ee_quat, ee_jacobian=ee_jacobian
        )
        return info_dict

    def reset(self) -> tuple[torch.Tensor, dict]:
        self._reset(torch.arange(self.num_envs, device=gs.device))
        obs = self.obs()
        obs = {k: v.cpu().numpy()[0] for k, v in obs.items()}
        info = self.info()
        info = {k: v.cpu().numpy()[0] for k, v in info.items()}
        return obs, info

    def step(self, actions: torch.Tensor | NDArray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # update step count
        self.steps += 1

        # apply actions and step simulation
        actions = torch.tensor(actions, dtype=gs.tc_float, device=self._device)
        self._apply_force(actions)
        self.scene.step()

        # construct outputs
        obs = self.obs()
        obs = {k: v.cpu().numpy()[0] for k, v in obs.items()}
        reward = self.reward().cpu().numpy()[0]
        done = self.done().cpu().numpy()[0]
        info = self.info()
        info = {k: v.cpu().numpy()[0] for k, v in info.items()}

        # extra render
        if self.steps % 2 == 0:
            self.render_trace(obs, info)

        return obs, reward, done, info

    def render_trace(self, obs: dict[str, NDArray[np.floating]], info: dict):
        if not hasattr(self, "trace"):
            self.trace = [np.array([0.3, 0.0, 0.3])]
        new_pos = obs["ee_pos"].copy()
        new_pos[2] = 0.3  # project to surface
        self.trace.append(new_pos)
        F_ext = info.get("ee_task_F_ext", obs["F_ext"])
        hardness = np.abs(F_ext[2] - 5.0) * 5e-4
        self.scene.draw_debug_line(self.trace[-2], self.trace[-1], radius=hardness, color=(0.0, 0.0, 0.0, 1.0))
