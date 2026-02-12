"""Single Franka robot environment for contact tasks."""

import math

import genesis as gs
import torch

from franesis.envs.franka_core import FrankaCore


class FrankaEnv(FrankaCore):
    def __init__(
        self,
        show_viewer: bool = False,
        num_envs: int = 1,
        episode_length_s: float = 3.0,
        freq: int = 100,
        render: bool = False,
    ) -> None:
        self.num_envs = num_envs
        self.num_privileged_obs = None
        self.device = gs.device

        self.ctrl_dt = 1.0 / freq
        self.max_episode_length = math.ceil(episode_length_s / self.ctrl_dt)

        # Store config parameters
        self._visualize_camera = render

        # 1. setup scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.ctrl_dt, substeps=2),
            rigid_options=gs.options.RigidOptions(
                dt=self.ctrl_dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(num_envs))),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.ctrl_dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            profiling_options=gs.options.ProfilingOptions(show_FPS=False),
            show_viewer=show_viewer,
        )

        # 2. add ground
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # 3. init robot
        super().__init__(num_envs=self.num_envs, scene=self.scene, device=gs.device)

        # 4. setup cameras
        if self._visualize_camera:
            self.vis_cam = self.scene.add_camera(
                res=(1280, 720),
                pos=(1.5, 0.0, 0.2),
                lookat=(0.0, 0.0, 0.2),
                fov=60,
                GUI=self._visualize_camera,
                debug=True,
            )

        # build
        self.scene.build(n_envs=num_envs)
        # set pd gains (must be called after scene.build)
        self._set_pd_gains()

        # == init buffers ==
        self._init_buffers()
        self.reset()

    def _init_buffers(self) -> None:
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=gs.device)
        self.goal_pose = torch.zeros(self.num_envs, 7, device=gs.device)

    def reset_idx(self, envs_idx: torch.Tensor) -> None:
        if len(envs_idx) == 0:
            return
        self.episode_length_buf[envs_idx] = 0

        # reset robot
        self._reset_home(envs_idx)

    def reset(self) -> tuple[torch.Tensor, dict]:
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        return self.obs()

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # update time
        self.episode_length_buf += 1

        self._apply_force(actions)
        self.scene.step()

        # check termination
        env_reset_idx = self.is_episode_complete()
        if len(env_reset_idx) > 0:
            self.reset_idx(env_reset_idx)

        # get observations
        obs = self.obs()

        return obs, torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float), self.reset_buf, {}

    def is_episode_complete(self) -> torch.Tensor:
        time_out_buf = self.episode_length_buf > self.max_episode_length

        # check if the ee is in the valid position
        self.reset_buf = time_out_buf
        return self.reset_buf.nonzero(as_tuple=True)[0]

    def obs(self) -> tuple[torch.Tensor, dict]:
        ee_pos, ee_quat = self._get_ee_pose()
        obs_dict = {
            "q": self._get_q(),
            "dq": self._get_dq(),
            "ee_pos": ee_pos,
            "ee_quat": ee_quat,
        }
        return obs_dict

