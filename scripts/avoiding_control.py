

import argparse
import math

import torch
import genesis as gs


################################################################################
#                              Manipulator Wrapper                             #
################################################################################

class Manipulator:
    """
    Minimal Franka wrapper:
    - Controls 7 arm joints with joint velocity commands
    - Gripper joints remain at their current positions (not actively controlled)
    """

    def __init__(self, num_envs: int, scene: gs.Scene, device: str = "cpu") -> None:
        self._device = device
        self._scene = scene
        self._num_envs = num_envs

        # ------------------------- Load Franka model ------------------------- #
        material = gs.materials.Rigid()
        morph = gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
            pos=(0.0, 0.0, 0.0),
            quat=(1.0, 0.0, 0.0, 0.0),
        )
        self._robot_entity: gs.Entity = scene.add_entity(
            material=material, morph=morph
        )

        # Gripper open/close distances (only used for reset / home pose)
        self._gripper_open_dof = 0.04
        self._gripper_close_dof = 0.00

        # Initialize DOF indices, link handles, etc.
        self._init_indices()

        # Home pose: 7 arm DOFs + 2 gripper DOFs
        self._default_arm_q = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
        self._default_gripper_q = [0.04, 0.04]

    # ---------------------------------------------------------------------- #
    #                          Internal Initialization                       #
    # ---------------------------------------------------------------------- #

    def _init_indices(self):
        """Initialize DOF dimensions, indices and link handles."""
        self._arm_dof_dim = self._robot_entity.n_dofs - 2  # 7 arm joints
        self._gripper_dim = 2

        self._arm_dof_idx = torch.arange(
            self._arm_dof_dim, device=self._device
        )
        self._fingers_dof = torch.arange(
            self._arm_dof_dim,
            self._arm_dof_dim + self._gripper_dim,
            device=self._device,
        )

        # End-effector and finger links
        self._ee_link = self._robot_entity.get_link("hand")
        self._left_finger_link = self._robot_entity.get_link("left_finger")
        self._right_finger_link = self._robot_entity.get_link("right_finger")

    # ---------------------------------------------------------------------- #
    #                             Low-level Control                         #
    # ---------------------------------------------------------------------- #

    def set_pd_gains(self):
        """Set PD gains and torque limits (copied from GraspEnv)."""
        self._robot_entity.set_dofs_kp(
            torch.tensor(
                [4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100],
                dtype=gs.tc_float,
                device=self._device,
            )
        )
        self._robot_entity.set_dofs_kv(
            torch.tensor(
                [450, 450, 350, 350, 200, 200, 200, 10, 10],
                dtype=gs.tc_float,
                device=self._device,
            )
        )
        self._robot_entity.set_dofs_force_range(
            torch.tensor(
                [-87, -87, -87, -87, -12, -12, -12, -100, -100],
                dtype=gs.tc_float,
                device=self._device,
            ),
            torch.tensor(
                [87, 87, 87, 87, 12, 12, 12, 100, 100],
                dtype=gs.tc_float,
                device=self._device,
            ),
        )

    def reset_home(self, envs_idx: torch.Tensor | None = None):
        """Reset robot to home pose."""
        if envs_idx is None:
            envs_idx = torch.arange(self._num_envs, device=self._device)

        q = torch.tensor(
            self._default_arm_q + self._default_gripper_q,
            dtype=gs.tc_float,
            device=self._device,
        ).repeat(len(envs_idx), 1)
        self._robot_entity.set_qpos(q, envs_idx=envs_idx)

    def apply_velocity(self, qdot_arm: torch.Tensor):
        """
        Apply joint velocity control on the arm.

        Args:
            qdot_arm: [N, 7] arm joint velocities
        """
        full_vel = torch.zeros(
            (self._num_envs, self._robot_entity.n_dofs),
            device=self._device,
            dtype=gs.tc_float,
        )
        full_vel[:, : self._arm_dof_dim] = qdot_arm
        full_vel[:, self._fingers_dof] = 0.0  # keep gripper still

        self._robot_entity.control_dofs_velocity(velocity=full_vel)

    # ---------------------------------------------------------------------- #
    #                               State Access                             #
    # ---------------------------------------------------------------------- #

    def get_arm_q(self) -> torch.Tensor:
        """Return current 7 arm joint positions q: [N, 7]."""
        return self._robot_entity.get_dofs_position(dofs_idx_local=self._arm_dof_idx)

    def get_arm_dq(self) -> torch.Tensor:
        """Return current 7 arm joint velocities dq: [N, 7]."""
        return self._robot_entity.get_dofs_velocity(dofs_idx_local=self._arm_dof_idx)

    @property
    def ee_pose(self) -> torch.Tensor:
        """End-effector (hand link) pose [N, 7] = (xyz + quat)."""
        pos, quat = self._ee_link.get_pos(), self._ee_link.get_quat()
        return torch.cat([pos, quat], dim=-1)

    @property
    def center_finger_pose(self) -> torch.Tensor:
        """
        Pose of the center point between the two fingers: [N, 7] = (xyz + quat).
        Position is the midpoint, orientation is taken from the left finger.
        """
        left_pos, left_quat = (
            self._left_finger_link.get_pos(),
            self._left_finger_link.get_quat(),
        )
        right_pos, right_quat = (
            self._right_finger_link.get_pos(),
            self._right_finger_link.get_quat(),
        )
        center_pos = (left_pos + right_pos) / 2.0
        center_quat = left_quat
        return torch.cat([center_pos, center_quat], dim=-1)

    def get_jacobian_ee(self) -> torch.Tensor:
        """Return end-effector Jacobian [N, 6, n_dofs]."""
        return self._robot_entity.get_jacobian(link=self._ee_link)


################################################################################
#                          APF Obstacle-Avoiding Env                           #
################################################################################

class APFAvoidingEnv:
    """
    Minimal environment for control-only experiments:

    - Panda arm + ground + one spherical obstacle + one spherical goal
    - Control interface: step(qdot_arm) with joint velocity commands
    - Exposes:
        - goal_pos      : [N, 3]
        - obstacle_pos  : [N, 3]
        - robot state via env.robot
    """

    def __init__(self, num_envs: int = 1, show_viewer: bool = True) -> None:
        self.num_envs = num_envs
        self.device = gs.device

        # ============================ Time parameters ======================== #
        self.ctrl_dt = 0.02
        self.episode_length_s = 5.0
        self.max_episode_length = math.ceil(self.episode_length_s / self.ctrl_dt)

        # ============================ Task parameters ======================== #
        self.obstacle_radius = 0.10
        self.obstacle_margin = 0.05
        self.safe_radius = self.obstacle_radius + self.obstacle_margin
        self.success_threshold = 0.03  # success if EE is within 3 cm of goal

        # ============================ Build Scene ============================ #
        max_render_envs = min(self.num_envs, 10)
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.ctrl_dt, substeps=2),
            rigid_options=gs.options.RigidOptions(
                dt=self.ctrl_dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=list(range(max_render_envs))
            ),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(1.0 / self.ctrl_dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            profiling_options=gs.options.ProfilingOptions(show_FPS=False),
            renderer=gs.options.renderers.BatchRenderer(use_rasterizer=True),
            show_viewer=show_viewer,
        )

        # At least one camera is required when using BatchRenderer
        self.main_cam = self.scene.add_camera(
            res=(1280, 720),
            pos=(1.5, 0.0, 0.8),
            lookat=(0.4, 0.0, 0.3),
            fov=60,
            GUI=show_viewer,
        )

        # ------------------------------ Ground ------------------------------ #
        self.scene.add_entity(
            gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True)
        )

        # ------------------------------ Robot ------------------------------- #
        self.robot = Manipulator(
            num_envs=self.num_envs, scene=self.scene, device=self.device
        )

        # ------------------------------ Obstacle ---------------------------- #
        self.obstacle = self.scene.add_entity(
            gs.morphs.Sphere(
                radius=self.obstacle_radius,
                fixed=True,
                collision=True,
                batch_fixed_verts=True,
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(color=(1.0, 0.4, 0.0))
            ),
        )

        # ------------------------------- Goal -------------------------------- #
        self.goal = self.scene.add_entity(
            gs.morphs.Sphere(
                radius=0.02,
                fixed=True,
                collision=False,
                batch_fixed_verts=True,
            ),
            surface=gs.surfaces.Emission(color=(0.0, 1.0, 0.0)),
        )

        # ---------------------- Build + set PD gains ------------------------ #
        self.scene.build(n_envs=self.num_envs, env_spacing=(1.0, 1.0))
        self.robot.set_pd_gains()

        # -------------------------- State buffers --------------------------- #
        self.goal_pos = torch.zeros(
            self.num_envs, 3, device=self.device, dtype=gs.tc_float
        )
        self.obstacle_pos = torch.zeros(
            self.num_envs, 3, device=self.device, dtype=gs.tc_float
        )
        self.episode_length_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=gs.tc_int
        )

        self.reset()

    # ---------------------------------------------------------------------- #
    #                     Sampling Goal / Obstacle Positions                 #
    # ---------------------------------------------------------------------- #

    def _sample_goal(self, batch_size: int) -> torch.Tensor:
        """
        Sample goal positions in a reachable forward workspace region.
        Example ranges (can be tuned):
            x ∈ [0.35, 0.65]
            y ∈ [-0.25, 0.25]
            z ∈ [0.10, 0.30]
        """
        x = torch.rand(batch_size, device=self.device, dtype=gs.tc_float) * 0.3 + 0.35
        y = (torch.rand(batch_size, device=self.device, dtype=gs.tc_float) - 0.5) * 0.5
        z = torch.rand(batch_size, device=self.device, dtype=gs.tc_float) * 0.2 + 0.1
        return torch.stack([x, y, z], dim=-1)

    def _sample_obstacle(self, batch_size: int, goal: torch.Tensor) -> torch.Tensor:
        """
        Sample obstacle positions around the midpoint between base and goal.
        This places the obstacle roughly "in the way" of the straight line path.
        """
        base = torch.zeros(batch_size, 3, device=self.device, dtype=gs.tc_float)
        mid = 0.5 * (base + goal)

        radius = self.obstacle_radius
        lateral_offset = (
            torch.rand(batch_size, device=self.device, dtype=gs.tc_float)
            * (radius * 0.8)
        )
        side = torch.where(
            torch.rand(batch_size, device=self.device) > 0.5,
            torch.ones(batch_size, device=self.device),
            -torch.ones(batch_size, device=self.device),
        )

        obs_x = mid[:, 0] + (
            torch.rand(batch_size, device=self.device, dtype=gs.tc_float) - 0.5
        ) * 0.05
        obs_y = mid[:, 1] + side * lateral_offset
        obs_z = mid[:, 2] + (
            torch.rand(batch_size, device=self.device, dtype=gs.tc_float) - 0.5
        ) * 0.5

        obs = torch.stack([obs_x, obs_y, obs_z], dim=-1)

        # Clamp into a reasonable 3D box
        obs[:, 0] = torch.clamp(obs[:, 0], 0.1, 0.8)
        obs[:, 1] = torch.clamp(obs[:, 1], -0.4, 0.4)
        obs[:, 2] = torch.clamp(obs[:, 2], 0.5, 1.5)
        return obs

    # ---------------------------------------------------------------------- #
    #                                Reset / Step                            #
    # ---------------------------------------------------------------------- #

    def reset(self):
        """
        Reset one episode:
        - Reset robot to home pose
        - Sample new goal and obstacle
        """
        self.episode_length_buf[:] = 0

        # Reset robot
        self.robot.reset_home()

        # Sample goal and obstacle
        goal = self._sample_goal(self.num_envs)
        obs = self._sample_obstacle(self.num_envs, goal)

        self.goal_pos[:] = goal
        self.obstacle_pos[:] = obs

        self.goal.set_pos(goal)
        self.obstacle.set_pos(obs)

        return {}, {}  # observation dict is unused here

    def step(self, qdot_arm: torch.Tensor):
        """
        Apply joint velocity and advance simulation by one step.

        Args:
            qdot_arm: [N, 7] arm joint velocities

        Returns:
            info: dict with flags:
                - is_success   : [N] float (1.0 if success else 0.0)
                - is_collision : [N] float (1.0 if collision else 0.0)
                - time_outs    : [N] float (1.0 if timeout else 0.0)
        """
        self.episode_length_buf += 1

        # Apply joint velocity
        self.robot.apply_velocity(qdot_arm)

        # Advance physics
        self.scene.step()

        # Check termination conditions
        info = self._check_done()
        return info

    def _check_done(self):
        """Compute success / collision / timeout flags."""
        ee_pos = self.robot.center_finger_pose[:, :3]
        goal_dist = torch.norm(ee_pos - self.goal_pos, dim=-1)
        obs_dist = torch.norm(ee_pos - self.obstacle_pos, dim=-1)

        time_out_buf = self.episode_length_buf > self.max_episode_length
        success_buf = goal_dist < self.success_threshold
        collision_buf = obs_dist < self.obstacle_radius

        info = {
            "is_success": success_buf.to(dtype=gs.tc_float),
            "is_collision": collision_buf.to(dtype=gs.tc_float),
            "time_outs": time_out_buf.to(dtype=gs.tc_float),
        }
        return info


################################################################################
#                           APF + Nullspace Controller                         #
################################################################################

def apf_control_action(
    env: APFAvoidingEnv,
    k_att: float = 2.0,
    k_rep: float = 0.3,
    d0: float | None = None,
    max_ee_vel: float = 0.4,
    max_joint_vel: float = 1.0,
    k_posture: float = 0.2,   # nullspace posture gain
) -> torch.Tensor:
    """
    Compute one control step using:

    - Artificial Potential Field (APF) in task (EE position) space:
        F = F_att + F_rep
    - Damped Least Squares (DLS) inverse Jacobian:
        qdot_task = J^T (J J^T + λ^2 I)^{-1} v_des
    - Nullspace posture control towards home pose:
        qdot = qdot_task + (I - J^+ J) * k_posture * (q0 - q)

    Returns:
        qdot_arm: [num_envs, 7] arm joint velocities
    """
    device = env.device
    num_envs = env.num_envs

    # ===================== EE / goal / obstacle positions ================== #
    ee_pos   = env.robot.center_finger_pose[:, :3]   # [N, 3]
    goal_pos = env.goal_pos                          # [N, 3]
    obs_pos  = env.obstacle_pos                      # [N, 3]

    # ============================ Attractive term ========================== #
    vec_g = goal_pos - ee_pos                        # [N, 3], EE -> goal
    F_att = k_att * vec_g                            # [N, 3]

    # ============================ Repulsive term ========================== #
    if d0 is None:
        d0 = float(env.safe_radius * 2.0)            # influence radius

    vec_o  = ee_pos - obs_pos                        # [N, 3], obstacle -> EE
    dist_o = torch.norm(vec_o, dim=-1, keepdim=True) + 1e-6  # [N, 1]

    # Only apply repulsion when inside d0
    mask = (dist_o < d0).to(dtype=gs.tc_float)       # [N, 1]
    coef = k_rep * (1.0 / dist_o - 1.0 / d0) * (1.0 / (dist_o * dist_o))
    F_rep = mask * coef * vec_o                      # [N, 3]

    # ========================= Total force & EE velocity =================== #
    F = F_att + F_rep                                # [N, 3]

    norm_F = torch.norm(F, dim=-1, keepdim=True) + 1e-6
    scale  = torch.clamp(max_ee_vel / norm_F, max=1.0)
    v_des  = F * scale                               # [N, 3]

    # ========================= DLS inverse kinematics ====================== #
    jac = env.robot.get_jacobian_ee()                # [N, 6, dofs]
    J   = jac[:, :3, : env.robot._arm_dof_dim]       # [N, 3, 7] (linear part)
    J_T = J.transpose(1, 2)                          # [N, 7, 3]

    lam  = 0.01
    eye3 = torch.eye(3, device=device, dtype=gs.tc_float).unsqueeze(0).repeat(num_envs, 1, 1)
    A    = J @ J_T + (lam**2) * eye3                 # [N, 3, 3]
    A_inv = torch.inverse(A)                         # [N, 3, 3]

    v    = v_des.unsqueeze(-1)                       # [N, 3, 1]
    x    = A_inv @ v                                 # [N, 3, 1]
    qdot_task = (J_T @ x).squeeze(-1)                # [N, 7]

    # ========================= Nullspace posture control =================== #
    # Desired posture q0: home pose (7 arm joints)
    q  = env.robot.get_arm_q()                       # [N, 7]
    q0 = torch.tensor(
        env.robot._default_arm_q,
        device=device,
        dtype=gs.tc_float,
    ).unsqueeze(0).repeat(num_envs, 1)               # [N, 7]

    # Pseudoinverse J^+ = J^T (J J^T + λ^2 I)^(-1)
    J_pinv = J_T @ A_inv                             # [N, 7, 3]
    eye7   = torch.eye(env.robot._arm_dof_dim, device=device, dtype=gs.tc_float).unsqueeze(0).repeat(num_envs, 1, 1)
    N      = eye7 - J_pinv @ J                       # [N, 7, 7]

    # Posture term (pull q towards q0), projected into nullspace
    qdot_posture = k_posture * (q0 - q)              # [N, 7]
    qdot_posture = (N @ qdot_posture.unsqueeze(-1)).squeeze(-1)  # [N, 7]

    # ========================= Combine & clamp ============================= #
    qdot = qdot_task + qdot_posture                  # [N, 7]
    qdot = torch.clamp(qdot, -max_joint_vel, max_joint_vel)

    return qdot


################################################################################
#                               Main Run Loop                                  #
################################################################################

def run_control_demo(
    num_episodes: int = 5,
    max_steps_per_ep: int = 300,
    vis: bool = True,
):
    """
    Run several episodes of the APF obstacle-avoidance demo.
    """
    gs.init(logging_level="warning", precision="32")

    env = APFAvoidingEnv(num_envs=1, show_viewer=vis)

    for ep in range(num_episodes):
        env.reset()
        print(f"\n=== Episode {ep} ===")
        for step in range(max_steps_per_ep):
            # Compute one control step
            qdot = apf_control_action(env)
            info = env.step(qdot)

            is_success = info["is_success"][0].item()
            is_collision = info["is_collision"][0].item()
            time_out = info["time_outs"][0].item()

            if is_success > 0.5:
                print(f"  step {step}: SUCCESS ")
                break
            if is_collision > 0.5:
                print(f"  step {step}: COLLISION ")
                break
            if time_out > 0.5:
                print(f"  step {step}: TIMEOUT ")
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument(
        "-v", "--vis",
        action="store_true",
        default=True,
        help="Enable Genesis viewer",
    )
    args = parser.parse_args()

    run_control_demo(
        num_episodes=args.episodes,
        max_steps_per_ep=args.steps,
        vis=args.vis,
    )


if __name__ == "__main__":
    main()
