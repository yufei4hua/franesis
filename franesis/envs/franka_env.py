"""Single Franka robot environment for contact tasks."""

import math

import genesis as gs
import torch
from numpy.typing import NDArray

from franesis.envs.franka_core import FrankaCore


class FrankaEnv(FrankaCore):
    def __init__(self, episode_length_s: float = 10.0, freq: int = 100, render: bool = True, device: str = "cuda"):
        super().__init__(num_envs=1, freq=freq, render=render, device=device)
        self.max_episode_length = math.ceil(episode_length_s / self.ctrl_dt)

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
