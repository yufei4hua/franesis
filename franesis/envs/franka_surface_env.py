"""Single Franka robot environment for contact tasks."""

import math

import genesis as gs
import numpy as np
import torch
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from franesis.envs.franka_core import FrankaCore


class FrankaSurfaceEnv(FrankaCore):
    def __init__(
        self,
        episode_length_s: float = 3.0,
        freq: int = 100,
        substeps: int = 10,
        render: bool = True,
        device: str = "cuda",
    ):
        super().__init__(num_envs=1, freq=freq, substeps=substeps, f_ext_lpf_alpha=0.5, render=render, device=device)
        self.max_episode_length = math.ceil(episode_length_s / self.ctrl_dt)
        self._default_arm_q = [0.13473345, -0.80271834, -0.13701877, -2.83875, -0.12417492, 2.0410793, 0.85577106]

    def _add_task_entities(self) -> None:
        # Cylinder parameters
        radius = 0.6
        length = 0.4  # cylinder half-length along its axis

        # Place it in front of robot
        cyl_pos = (0.3, 0.0, 0.3 - radius)

        # Rotate cylinder
        quat_xyzw = R.from_euler("y", 90, degrees=True).as_quat()  # (x,y,z,w)
        cyl_quat_wxyz = (float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2]))

        cyl_material = gs.materials.Rigid(friction=0.02, coup_softness=0.02, coup_restitution=0.0)

        cyl_morph = gs.morphs.Cylinder(
            pos=cyl_pos,
            quat=cyl_quat_wxyz,
            radius=radius,
            height=length,
            fixed=True,
            collision=True,
            visualization=True,
        )

        cyl_surface = gs.surfaces.Smooth()
        self.surface: gs.Entity = self.scene.add_entity(material=cyl_material, morph=cyl_morph, surface=cyl_surface)

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
        hardness = np.abs(obs["F_ext"][2] - 30.0) * 4e-4
        self.scene.draw_debug_line(self.trace[-2], self.trace[-1], radius=hardness, color=(0.0, 0.0, 0.0, 1.0))
