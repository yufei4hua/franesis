"""Single Franka robot environment for contact tasks."""

import math

import genesis as gs
import torch
from numpy.typing import NDArray

from franesis.envs.franka_core import FrankaCore


class FrankaSphereEnv(FrankaCore):
    def __init__(
        self,
        episode_length_s: float = 3.0,
        freq: int = 100,
        substeps: int = 10,
        render: bool = True,
        device: str = "cuda",
    ):
        super().__init__(num_envs=1, freq=freq, substeps=substeps, render=render, device=device)
        self.max_episode_length = math.ceil(episode_length_s / self.ctrl_dt)
        self._default_arm_q = [0.13473345, -0.80271834, -0.13701877, -2.83875, -0.12417492, 2.0410793, 0.85577106]

    def _add_task_entities(self) -> None:
        # --- Sphere dimensions (meters) ---
        radius = 0.1
        sphere_pos = (0.3, 0.0, 0.1 + radius)  # Center position on z=0 plane with radius offset

        # Rigid material with friction
        box_material = gs.materials.Rigid(friction=0.02, coup_softness=0.02, coup_restitution=0.0)

        # Fixed sphere: baselink fixed => will not be pushed away :contentReference[oaicite:3]{index=3}
        box_morph = gs.morphs.Sphere(pos=sphere_pos, radius=radius, fixed=True, collision=True, visualization=True)

        # Optional: surface only affects rendering by default; physics is in material
        box_surface = gs.surfaces.Default()

        self.box: gs.Entity = self.scene.add_entity(material=box_material, morph=box_morph, surface=box_surface)

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
        info_dict = {"ee_jacobian": self._get_jacobian_ee()}
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

        return obs, reward, done, info
