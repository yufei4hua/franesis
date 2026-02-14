"""Franka Core Env for low-level control and state access."""

import genesis as gs
import torch
from genesis.utils.geom import transform_by_quat


class FrankaCore:
    def __init__(
        self,
        num_envs: int,
        freq: int = 100,
        substeps: int = 10,
        f_ext_lpf_alpha: float = 1.0,
        render: bool = False,
        ee_name: str = "tool_tip",
        device: str = "cpu",
    ) -> None:
        self._device = device
        self.num_envs = num_envs
        self.device = gs.device
        self.freq = freq
        self.ctrl_dt = 1.0 / freq
        self.render = render
        self.ee_name = ee_name
        self.steps = torch.zeros((self.num_envs,), device=self._device, dtype=torch.int32)
        self.f_ext_lpf_alpha = f_ext_lpf_alpha

        gs.init(backend=gs.gpu)

        # 1. setup scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.ctrl_dt, substeps=substeps),
            rigid_options=gs.options.RigidOptions(
                dt=self.ctrl_dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                constraint_timeconst=0.1,
                iterations=100,
                noslip_iterations=0,
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=list(range(self.num_envs)),
                contact_force_scale=0.01,  # m/N
            ),
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
        material = gs.materials.Rigid(friction=0.02, coup_softness=0.02, coup_restitution=0.0)
        morph = gs.morphs.MJCF(
            file="franesis/envs/franka_emika_panda/panda_cylinder.xml", pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0)
        )
        self.robot: gs.Entity = self.scene.add_entity(material=material, morph=morph)
        self._add_task_entities()

        # 4. build scene
        self.scene.build(n_envs=self.num_envs)

        # setup indices and controller pd gains (must be called after scene.build)
        self._default_arm_q = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
        self._default_gripper_q = [0.04, 0.04]
        self._init_indices()
        self._set_pd_gains()
        self.applied_force = torch.zeros((self.num_envs, self.robot.n_dofs), device=self._device, dtype=gs.tc_float)
        self._f_ext_filtered = torch.zeros((self.num_envs, 6), device=self._device, dtype=gs.tc_float)

    def _init_indices(self):
        self._arm_dof_dim = min(7, self.robot.n_dofs)
        self._gripper_dim = max(0, self.robot.n_dofs - self._arm_dof_dim)

        # Match default qpos dimensions to the current robot model (with/without gripper).
        self._default_arm_q = self._default_arm_q[: self._arm_dof_dim]
        self._default_gripper_q = self._default_gripper_q[: self._gripper_dim]

        self._arm_dof_idx = torch.arange(self._arm_dof_dim, device=self._device)
        self._fingers_dof = torch.arange(self._arm_dof_dim, self._arm_dof_dim + self._gripper_dim, device=self._device)

        self._ee_link = self.robot.get_link(self.ee_name)

    def _add_task_entities(self):
        pass

    def _set_pd_gains(self):
        kp = [0] * self.robot.n_dofs
        kv = [0] * self.robot.n_dofs

        fr_min = [-87, -87, -87, -87, -12, -12, -12] + ([-100] * self._gripper_dim)
        fr_max = [87, 87, 87, 87, 12, 12, 12] + ([100] * self._gripper_dim)

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
        self._f_ext_filtered[mask] = 0.0

    def _apply_force(self, tau: torch.Tensor):
        full_tau = torch.zeros((self.num_envs, self.robot.n_dofs), device=self._device, dtype=gs.tc_float)
        full_tau[:, : self._arm_dof_dim] = tau
        full_tau[:, self._fingers_dof] = 0.0
        self.applied_force = full_tau
        self.robot.control_dofs_force(force=full_tau)

    def _get_q(self) -> torch.Tensor:
        return self.robot.get_dofs_position(dofs_idx_local=self._arm_dof_idx)

    def _get_dq(self) -> torch.Tensor:
        return self.robot.get_dofs_velocity(dofs_idx_local=self._arm_dof_idx)

    def _get_ee_pose(self) -> torch.Tensor:
        pos, quat = self._ee_link.get_pos(), self._ee_link.get_quat()  # (w, x, y, z)
        return pos, torch.roll(quat, shifts=-1, dims=-1)  # (x, y, z, w)
    
    def _get_jacobian_ee(self) -> torch.Tensor:
        return self.robot.get_jacobian(link=self._ee_link)

    def _get_ee_F_ext(self) -> torch.Tensor:
        F_ext_raw = torch.zeros((1, 6), device=self._device, dtype=gs.tc_float)
        a = self.f_ext_lpf_alpha

        Fw, Tw, _, _, _ = self._get_cylinder_contact_wrench("sensor")
        F_ext_raw[0, :3] = Fw
        F_ext_raw[0, 3:] = Tw

        # omit outliers
        F_ext_raw = torch.where(F_ext_raw.abs() < 1e-3, (1.0 - a) * self._f_ext_filtered, F_ext_raw)

        # low pass filter
        self._f_ext_filtered = a * F_ext_raw + (1.0 - a) * self._f_ext_filtered

        return self._f_ext_filtered

    def _get_tau_ext(self) -> torch.Tensor:
        J = self._get_jacobian_ee()  # (B, 6, n_dofs)
        F_ext = self._get_ee_F_ext()  # (B, 6)
        tau_ext = torch.bmm(J.transpose(1, 2), F_ext.unsqueeze(-1)).squeeze(-1)  # (B, n_dofs)
        return tau_ext

    def _get_cylinder_contact_wrench(
        self, paddle_link_name: str = "sensor"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Link-based contact wrench extraction (no geom / idx_global dependency).

        We only integrate contacts where one side is the link named `paddle_link_name`.
        Returns:
            F_world: (3,) net force in world frame
            T_world: (3,) net torque in world frame about link origin
            F_local: (3,) net force in link local frame
            T_local: (3,) net torque in link local frame
            n_contacts: int, number of contributing contact points
        """
        # Get the link and its indices
        link_paddle = self.robot.get_link(paddle_link_name)
        link_idx_local = link_paddle.idx_local
        link_idx_global = self.robot.link_start + link_idx_local

        # Link pose in world frame
        p_link = link_paddle.get_pos().squeeze()  # [3]
        q_link = link_paddle.get_quat().squeeze()  # [4] (x, y, z, w)
        q_inv = torch.tensor([-q_link[0], -q_link[1], -q_link[2], q_link[3]], device=gs.device, dtype=gs.tc_float)

        # All contacts involving this robot
        c = self.robot.get_contacts()

        mask_a = c["link_a"] == link_idx_global
        mask_b = c["link_b"] == link_idx_global

        Fw = torch.zeros(3, device=gs.device, dtype=gs.tc_float)
        Tw = torch.zeros(3, device=gs.device, dtype=gs.tc_float)

        # Contacts where this link is on side A
        if mask_a.any():
            Fa = c["force_a"][mask_a]  # (na, 3)
            Pa = c["position"][mask_a]  # (na, 3)
            Fw += Fa.sum(dim=0)
            Tw += torch.cross(Pa - p_link, Fa).sum(dim=0)

        # Contacts where this link is on side B
        if mask_b.any():
            Fb = c["force_b"][mask_b]
            Pb = c["position"][mask_b]
            Fw += Fb.sum(dim=0)
            Tw += torch.cross(Pb - p_link, Fb).sum(dim=0)

        # NOTE: order is transform_by_quat(vector, quat)
        Fl = transform_by_quat(Fw, q_inv)
        Tl = transform_by_quat(Tw, q_inv)
        n = int(mask_a.sum().item() + mask_b.sum().item())
        return Fw, Tw, Fl, Tl, n
