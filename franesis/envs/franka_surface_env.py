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
        # Cylinder parameters
        self.radius = 0.6
        self.length = 0.4
        self.center = (0.3, 0.0, 0.3 - self.radius)

        super().__init__(num_envs=1, freq=freq, substeps=substeps, f_ext_lpf_alpha=0.5, render=render, device=device)
        self.max_episode_length = math.ceil(episode_length_s / self.ctrl_dt)
        self._default_arm_q = [0.13473345, -0.80271834, -0.13701877, -2.83875, -0.12417492, 2.0410793, 0.85577106]

    def _add_task_entities(self) -> None:
        # Rotate cylinder
        quat_xyzw = R.from_euler("y", 90, degrees=True).as_quat()  # (x,y,z,w)
        cyl_quat_wxyz = (float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2]))

        cyl_material = gs.materials.Rigid(friction=0.02, coup_softness=0.02, coup_restitution=0.0)

        cyl_morph = gs.morphs.Cylinder(
            pos=self.center,
            quat=cyl_quat_wxyz,
            radius=self.radius,
            height=self.length,
            fixed=True,
            collision=True,
            visualization=True,
        )

        cyl_surface = gs.surfaces.Smooth()
        self.surface: gs.Entity = self.scene.add_entity(material=cyl_material, morph=cyl_morph, surface=cyl_surface)

    @staticmethod
    def _task_jocobians(
        center: tuple[float, float, float],
        cylinder_radius: float,
        ee_pos: torch.Tensor,
        ee_quat: torch.Tensor,
        ee_jacobian: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # 0. convert to numpy and assume single env
        ee_pos = ee_pos.detach().cpu().numpy()[0]
        ee_quat = ee_quat.detach().cpu().numpy()[0]
        J = ee_jacobian.detach().cpu().numpy()[0]
        Jv = J[0:3, :]  # linear
        Jw = J[3:6, :]  # angular
        R_cyl = cylinder_radius

        # 1. task coordinate pos
        yr = ee_pos[1] - center[1]
        zr = ee_pos[2] - center[2]
        r = np.sqrt(yr**2 + zr**2)
        alpha = np.arctan2(yr, zr)
        s = R_cyl * alpha
        ee_task_pos = np.array(
            [
                ee_pos[0] - center[0],  # x
                s - center[1],  # arc length
                r - R_cyl,  # radial distance
            ]
        )

        # 2. task coordinate rotation
        R_act = R.from_quat(ee_quat).as_matrix()  # world_R_ee
        x_t = np.array([1.0, 0.0, 0.0])
        z_t = np.array([0.0, yr, zr]) / r
        y_t = np.cross(z_t, x_t)
        y_t = y_t / np.linalg.norm(y_t)
        R_w_t = np.column_stack([x_t, y_t, z_t]) # world2tangent
        R_task = R_w_t.T @ R_act
        ee_task_quat = R.from_matrix(R_task).as_quat()

        # 3. task position Jacobian
        d = yr**2 + zr**2
        inv_d = 1.0 / max(d, 1e-9)
        inv_r = 1.0 / r

        A_v = np.zeros((3, 3))
        A_v[0, 0] = 1.0
        A_v[1, 1] = R_cyl * zr * inv_d
        A_v[1, 2] = -R_cyl * yr * inv_d
        A_v[2, 1] = yr * inv_r
        A_v[2, 2] = zr * inv_r

        Jv_task = A_v @ Jv

        # 4. task rotation Jacobian
        Jw_task = R_w_t.T @ Jw

        # 5. construct Jacobians
        J_task = np.vstack([Jv_task, Jw_task])
        J_force = J_task[2:3, :]
        J_motion = np.vstack([J_task[0:2, :], J_task[3:6, :]])
        U, S, Vh = np.linalg.svd(J_task, full_matrices=True)
        J_null = Vh[-1:, :]
        J_null = J_null / max(np.linalg.norm(J_null), 1e-12)

        return {
            "ee_jacobian": torch.from_numpy(J).unsqueeze(0),
            "ee_task_pos": torch.from_numpy(ee_task_pos).unsqueeze(0),
            "ee_task_quat": torch.from_numpy(ee_task_quat).unsqueeze(0),
            "ee_jacobian_task": torch.from_numpy(J_task).unsqueeze(0),
            "ee_jacobian_force": torch.from_numpy(J_force).unsqueeze(0),
            "ee_jacobian_motion": torch.from_numpy(J_motion).unsqueeze(0),
            "ee_jacobian_null": torch.from_numpy(J_null).unsqueeze(0),
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
            center=self.center,
            cylinder_radius=self.radius,
            ee_pos=ee_pos,
            ee_quat=ee_quat,
            ee_jacobian=ee_jacobian,
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
        new_pos = np.asarray(obs["ee_pos"], dtype=float).copy()

        # Cylinder axis is along x: project only in yz plane by scaling with R / r.
        yz = new_pos[1:3] - np.array([self.center[1], self.center[2]], dtype=float)
        r = np.linalg.norm(yz)
        scale = float(self.radius) / max(float(r), 1e-9)
        new_pos[1:3] = np.array([self.center[1], self.center[2]], dtype=float) + yz * scale

        self.trace.append(new_pos)
        hardness = np.abs(obs["F_ext"][2] - 30.0) * 4e-4
        self.scene.draw_debug_line(self.trace[-2], self.trace[-1], radius=hardness, color=(0.0, 0.0, 0.0, 1.0))
