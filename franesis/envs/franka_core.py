"""Franka Core Env for low-level control and state access."""

import genesis as gs
import torch


class FrankaCore:
    def __init__(
        self, num_envs: int, episode_length_s: float = 3.0, freq: int = 100, render: bool = False, device: str = "cpu"
    ) -> None:
        self._device = device
        self.num_envs = num_envs
        self.device = gs.device
        self.ctrl_dt = 1.0 / freq
        self.render = render
        self.steps = torch.zeros((self.num_envs,), device=self._device, dtype=torch.int32)

        gs.init(backend=gs.gpu)

        # 1. setup scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.ctrl_dt, substeps=2),
            rigid_options=gs.options.RigidOptions(
                dt=self.ctrl_dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(self.num_envs))),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.ctrl_dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            profiling_options=gs.options.ProfilingOptions(show_FPS=False),
            show_viewer=render,
        )

        # 2. add ground
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # 3. add Franka robot
        material = gs.materials.Rigid()
        morph = gs.morphs.MJCF(
            file="franesis/envs/franka_emika_panda/panda_cylinder.xml", pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0)
        )
        self.robot: gs.Entity = self.scene.add_entity(material=material, morph=morph)

        # 4. setup cameras
        if self.render:
            self.vis_cam = self.scene.add_camera(
                res=(1280, 720), pos=(1.5, 0.0, 0.2), lookat=(0.0, 0.0, 0.2), fov=60, GUI=self.render, debug=True
            )

        # 5. build scene
        self.scene.build(n_envs=self.num_envs)

        # setup indices and controller pd gains (must be called after scene.build)
        self._default_arm_q = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
        self._default_gripper_q = [0.04, 0.04]
        self._init_indices()
        self._set_pd_gains()

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
            mask = torch.arange(self.num_envs, device=self._device)

        q = torch.tensor(self._default_arm_q + self._default_gripper_q, dtype=gs.tc_float, device=self._device).repeat(
            len(mask), 1
        )
        self.robot.set_qpos(q, envs_idx=mask)

    def _apply_force(self, tau: torch.Tensor):
        full_tau = torch.zeros((self.num_envs, self.robot.n_dofs), device=self._device, dtype=gs.tc_float)
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
