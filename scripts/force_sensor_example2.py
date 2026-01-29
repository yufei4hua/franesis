import argparse
import torch
import genesis as gs
from genesis.utils.geom import transform_by_quat

# ---------- Small utilities ----------
def ensure_2d(x: torch.Tensor) -> torch.Tensor:
    """Ensure tensor has shape [1, ...] instead of [...]."""
    return x.unsqueeze(0) if x.ndim == 1 else x


def debug_who_contacts(franka):
    """
    Return (idx, name, force_vector) of the link that currently has
    the largest net contact force. Helpful to see which link actually
    carries the maximum contact.
    """
    F = franka.get_links_net_contact_force()  # [B, n_links, 3] or [n_links, 3]
    if F.ndim == 3:
        F = F[0]
    norms = torch.linalg.norm(F, dim=-1)
    idx = int(torch.argmax(norms).item())
    try:
        name = getattr(franka.links[idx], "name", f"link_{idx}")
    except Exception:
        name = f"link_{idx}"
    return idx, name, F[idx]


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate for [x, y, z, w]."""
    x, y, z, w = q
    return torch.tensor([-x, -y, -z, w], device=gs.device, dtype=gs.tc_float)


def get_cylinder_contact_wrench(franka, paddle_link_name: str = "sensor"):
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
    link_paddle = franka.get_link(paddle_link_name)
    link_idx_local = link_paddle.idx_local
    link_idx_global = franka.link_start + link_idx_local

    # Link pose in world frame
    p_link = ensure_2d(link_paddle.get_pos())[0]  # [3]
    q_link = ensure_2d(link_paddle.get_quat())[0]  # [4] (x, y, z, w)
    q_inv = torch.tensor(
        [-q_link[0], -q_link[1], -q_link[2], q_link[3]],
        device=gs.device,
        dtype=gs.tc_float,
    )

    # All contacts involving this robot
    c = franka.get_contacts()

    mask_a = (c["link_a"] == link_idx_global)
    mask_b = (c["link_b"] == link_idx_global)

    Fw = torch.zeros(3, device=gs.device, dtype=gs.tc_float)
    Tw = torch.zeros(3, device=gs.device, dtype=gs.tc_float)

    # Contacts where this link is on side A
    if mask_a.any():
        Fa = c["force_a"][mask_a]   # (na, 3)
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


# ---------- Scene construction ----------
def build_scene(dt: float = 0.005, viewer: bool = True, force_scale: float = 0.005):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            substeps=2,
            gravity=(0.0, 0.0, -9.81),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.0, -0.6, 1.2),
            camera_lookat=(0.4, 0.0, 0.2),
            camera_fov=40,
            max_FPS=int(1.0 / dt),
        ),
        rigid_options=gs.options.RigidOptions(
            dt=dt,
            constraint_solver=gs.constraint_solver.Newton,
            enable_collision=True,
            enable_joint_limit=True,
        ),
        show_viewer=viewer,
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
    )

    # Ground plane (no contact visualization on plane side to reduce clutter)
    plane = scene.add_entity(gs.morphs.Plane(), visualize_contact=False)

    # Franka arm without hand (7 DOFs)
    franka = scene.add_entity(
        material=gs.materials.Rigid(),
        morph=gs.morphs.MJCF(file="xml/franka_emika_panda/panda_nohand.xml"),
        visualize_contact=True,
    )

    # Add a "table" (box) in front of the arm, sitting on the ground.
    # Size: 0.6 x 0.8 x 0.4 (x, y, z), so the top surface is at z = 0.4 m.
    table = scene.add_entity(
        gs.morphs.Box(
            size=(0.6, 0.8, 0.4),     # width (x), depth (y), height (z)
            pos=(0.60, 0.0, 0.2),     # center position: in front of base, on ground
        ),
        visualize_contact=False,       # show contact forces on the table as well
        material=gs.materials.Rigid(),
    )

    scene.build(n_envs=1)

    # Global visualization options (especially contact force arrows)
    try:
        scene.set_vis_options(
            gs.options.VisOptions(
                contact_forces=True,
                contact_force_scale=force_scale,  # m/N
            )
        )
    except Exception:
        # Some Genesis versions may not support this; safe to ignore
        pass

    # PD gains (if forces are too large, you can reduce these values)
    franka.set_dofs_kp(
        torch.tensor(
            [4500, 4500, 3500, 3500, 2000, 2000, 2000],
            dtype=gs.tc_float,
            device=gs.device,
        )
    )
    franka.set_dofs_kv(
        torch.tensor(
            [450, 450, 350, 350, 200, 200, 200],
            dtype=gs.tc_float,
            device=gs.device,
        )
    )
    franka.set_dofs_force_range(
        torch.tensor(
            [-87, -87, -87, -87, -20, -20, -20],
            dtype=gs.tc_float,
            device=gs.device,
        ),
        torch.tensor(
            [87, 87, 87, 87, 20, 20, 20],
            dtype=gs.tc_float,
            device=gs.device,
        ),
    )
    return scene, franka, plane, table


# ---------- Main logic ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--viewer", type=int, default=1)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument(
        "--slam_depth",
        type=float,
        default=-0.4,
        help="Target depth along -Z direction (m)."
    )
    parser.add_argument(
        "--arrow_scale",
        type=float,
        default=0.005,
        help="Visualization scale for contact forces (m/N)."
    )
    args = parser.parse_args()

    gs.init(backend=gs.gpu)
    scene, franka, plane, table = build_scene(
        dt=args.dt,
        viewer=bool(args.viewer),
        force_scale=args.arrow_scale,
    )

    # End-effector link (here using 'link7' in panda_nohand.xml)
    ee_link = franka.get_link("link7")
    ee_idx = ee_link.idx_local

    q_down = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=gs.device, dtype=gs.tc_float)
    dofs_idx = torch.arange(franka.n_dofs, device=gs.device)

    print(
        "\n>>> Contact-force debug on Franka with a table in front <<<\n"
        " - Plane contact visualization is OFF, but the table shows contact arrows.\n"
        " - We compute the net contact wrench on 'link7' (tip of the arm).\n"
        " - In the viewer, press 'V' to open the menu, make sure 'Contact forces: On'.\n"
        f" - You can tune arrow size via --arrow_scale (current: {args.arrow_scale}).\n"
    )

    for i in range(args.steps):
        # Target position: keep XY, move Z downward
        ee_pos = ensure_2d(ee_link.get_pos())
        target = ee_pos.clone()
        target[:, 2] = -abs(args.slam_depth)

        qpos = franka.inverse_kinematics(
            link=ee_link,
            pos=target,
            quat=q_down,
            dofs_idx_local=dofs_idx,
        )
        franka.control_dofs_position(position=qpos)
        scene.step()

        # (A) Net contact force per link (sanity check)
        F_links = franka.get_links_net_contact_force()
        if F_links.ndim == 3:
            F_links = F_links[0]
        F_ee = F_links[ee_idx]

        # (B) Net contact wrench on the chosen link (world/local)
        Fw, Tw, Fl, Tl, n = get_cylinder_contact_wrench(franka, "link7")

        # (C) Which link has the maximum net contact force?
        idx_max, name_max, F_max = debug_who_contacts(franka)

        if i % 10 == 0:
            print(
                f"[step {i:4d}]\n"
                f"  - Contact force on link7 (from get_links_net_contact_force):\n"
                f"      F_link7 = [{F_ee[0]: .3f}, {F_ee[1]: .3f}, {F_ee[2]: .3f}] N\n"
                f"  - Integrated wrench on link7 (world frame):\n"
                f"      F_world = ({Fw[0]: .3f}, {Fw[1]: .3f}, {Fw[2]: .3f}) N\n"
                f"      T_world = ({Tw[0]: .3f}, {Tw[1]: .3f}, {Tw[2]: .3f}) N·m\n"
                f"  - Integrated wrench on link7 (link-local frame):\n"
                f"      F_local = ({Fl[0]: .3f}, {Fl[1]: .3f}, {Fl[2]: .3f})\n"
                f"      T_local = ({Tl[0]: .3f}, {Tl[1]: .3f}, {Tl[2]: .3f})\n"
                f"  - Number of contact points on link7: {n}\n"
                f"  - Max-net-force link in the whole robot:\n"
                f"      name = {name_max}, idx = {idx_max}, F_max = {F_max.tolist()}\n"
            )


if __name__ == "__main__":
    main()
