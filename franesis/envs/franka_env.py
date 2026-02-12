"""Single Franka robot environment for contact tasks."""

import math

import genesis as gs
import torch

from franesis.envs.franka_core import FrankaCore


class FrankaEnv(FrankaCore):
    def __init__(self, episode_length_s: float = 5.0, freq: int = 100, render: bool = True, device: str = "cuda"):
        super().__init__(num_envs=1, freq=freq, render=render, device=device)
        self.max_episode_length = math.ceil(episode_length_s / self.ctrl_dt)

    def _reset(self, mask: torch.Tensor) -> None:
        if len(mask) == 0:
            return
        self.steps[mask] = 0

        # reset robot
        self._reset_home(mask)

    def obs(self) -> tuple[torch.Tensor, dict]:
        obs_dict = {
            "q": self._get_q(),
            "dq": self._get_dq(),
            "tau_ext": self._get_tau_ext(),
            "F_ext": self._get_ee_F_ext(),
        }
        return obs_dict

    def reward(self) -> torch.Tensor:
        return torch.zeros((self.num_envs,), device=self.device)

    def done(self) -> torch.Tensor:
        return self.steps > self.max_episode_length

    def info(self) -> dict:
        ee_pos, ee_quat = self._get_ee_pose()
        info_dict = {
            "ee_pos": ee_pos,
            "ee_quat": ee_quat,
            "ee_jacobian": self._get_jacobian_ee(),
        }
        return info_dict

    def reset(self) -> tuple[torch.Tensor, dict]:
        self._reset(torch.arange(self.num_envs, device=gs.device))
        init_info = self.info()
        return self.obs(), init_info

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # update step count
        self.steps += 1

        # apply actions and step simulation
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
