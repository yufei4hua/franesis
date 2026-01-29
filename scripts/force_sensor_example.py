import argparse
import torch
import genesis as gs


def build_scene(dt=0.005, viewer=True, force_scale=0.002):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            substeps=2,
            gravity=(0.0, 0.0, -10),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.2, -1.2, 0.8),
            camera_lookat=(0.0, 0.0, 0.0),
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

    # Add ground plane
    plane = scene.add_entity(
        gs.morphs.Plane(),
        visualize_contact=True,
    )

    rigid_material = gs.materials.Rigid(rho=500.0)

    box = scene.add_entity(
        gs.morphs.Box(
            size=(0.4, 0.4, 0.4),
            pos=(0.65, 0.0, 0.02),
        ),
        visualize_contact=True,  # Enable contact visualization on the box
        material=rigid_material,
    )

    scene.build(n_envs=1)

    return scene, box, plane


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--viewer", type=int, default=1)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument(
        "--arrow_scale",
        type=float,
        default=0.002,
        help="Contact force arrow scale (m/N)",
    )
    args = parser.parse_args()

    gs.init(backend=gs.gpu)
    scene, box, plane = build_scene(
        dt=args.dt,
        viewer=bool(args.viewer),
        force_scale=args.arrow_scale,
    )

    for i in range(args.steps):
        scene.step()

        # Read net contact force on the box link
        F_links = box.get_links_net_contact_force()
        if F_links.ndim == 3:
            F_links = F_links[0]
        # The box only has one link (idx = 0)
        F_box = F_links[0]

        if i % 20 == 0:
            print(
                f"step {i:4d} | Contact Force on box = "
                f"({F_box[0]:.3f}, {F_box[1]:.3f}, {F_box[2]:.3f}) N"
            )



if __name__ == "__main__":
    main()
