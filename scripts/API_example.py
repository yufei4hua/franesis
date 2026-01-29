#### 0. IMPORTS ####
# You only need:
# - genesis: the simulator
# - torch: because Genesis uses PyTorch tensors internally
import genesis as gs
import torch


#### 1. HELPER: squeeze_env_dim(x) ####
# Purpose:
#   In Genesis you can have multiple parallel environments (n_envs > 1).
#   In that case arrays often have shape like (n_envs, ...).
#   Here you said you only use ONE environment.
#
# Behavior:
#   - If a tensor has shape (1, N) or (1, N, 3), we remove the leading "1" dimension.
#   - If it does NOT start with 1, we leave it unchanged.
#
# When to use:
#   Use this on outputs that *might* have an env dimension: get_dofs_position, get_mass_mat, etc.
def squeeze_env_dim(x):
    if isinstance(x, torch.Tensor) and x.ndim >= 2 and x.shape[0] == 1:
        return x[0]
    return x


#### 2. MAIN FUNCTION: print_franka_dynamics(franka) ####
# Input:
#   franka: a RigidEntity object returned by scene.add_entity(...)
#
# Goal:
#   Show "student API" examples on how to query:
#   - Total robot mass
#   - Link inertial parameters (mass, COM, inertia tensor)
#   - Link COM in world frame
#   - Joint states (q, dq, tau)
#   - Joint gains / damping / armature
#   - Joint-space mass matrix M(q)
def print_franka_dynamics(franka):
    #### 2.1 Total mass of the whole robot ####
    # API: franka.get_mass()
    # Returns:
    #   A scalar: sum of the inertial masses of all links.
    print("\n================ Total mass of Franka ================")
    print("total mass:", franka.get_mass())

    #### 2.2 Link-level inertial parameters (from the MJCF model) ####
    # Attributes on each link:
    #   link.inertial_mass : mass (kg)
    #   link.inertial_pos  : COM position relative to the link frame (3D)
    #   link.inertial_quat : orientation of the inertial frame (quaternion)
    #   link.inertial_i    : 3x3 inertia tensor in the inertial frame
    #
    # Use when:
    #   - You want to inspect geometry / mass distribution of each rigid body.
    #   - You want to build your own dynamics or energy calculations per link.
    print("\n================ Link-level inertial parameters ================")
    print(f"Number of links: {franka.n_links}")
    for i, link in enumerate(franka.links):
        print(f"\n--- Link {i}: {link.name} ---")
        print(f"  inertial_mass (kg): {link.inertial_mass}")
        print(f"  inertial_pos  (COM offset in link frame): {link.inertial_pos}")
        print(f"  inertial_quat (inertial frame orientation): {link.inertial_quat}")
        print("  inertial_i    (inertia tensor in inertial frame):")
        print(link.inertial_i)

    #### 2.3 All link masses as a tensor ####
    # API: franka.get_links_inertial_mass()
    # Shape:
    #   - (n_links,) if you have 1 environment
    #   - (n_envs, n_links) if you have multiple environments
    #
    # Typical usage:
    #   - Summing to check total mass
    #   - Using as weights in link-level cost/reward terms
    print("\n================ All link masses (get_links_inertial_mass) ================")
    link_masses = squeeze_env_dim(franka.get_links_inertial_mass())
    print("shape:", tuple(link_masses.shape))
    print(link_masses)

    #### 2.4 Link centers of mass in WORLD frame ####
    # API: franka.get_links_pos(ref="link_com")
    # Arguments:
    #   ref="link_com"  -> return positions of each link's center of mass (CoM).
    #
    # Shape:
    #   - (n_links, 3)   for single env
    #   - (n_envs, n_links, 3) for multi env
    #
    # Use when:
    #   - You want to visualize or log CoM trajectories of each link.
    #   - You want to compute distances from link CoMs to obstacles.
    print("\n================ Link centers of mass in world frame ================")
    com_world = squeeze_env_dim(franka.get_links_pos(ref="link_com"))
    print("COM world pos shape:", tuple(com_world.shape))
    print(com_world)

    #### 2.5 Joint states: q (position), dq (velocity), tau (force/torque) ####
    # APIs:
    #   franka.get_dofs_position()  -> joint positions q
    #   franka.get_dofs_velocity()  -> joint velocities dq
    #   franka.get_dofs_force()     -> total joint forces/torques tau
    #
    # Shape:
    #   - (n_dofs,) or (1, n_dofs)
    #
    # Use when:
    #   - Building observations for RL / control
    #   - Logging trajectories
    #   - Computing power = tau * dq for energy-based costs
    print("\n================ DOF states (q, dq, tau) ================")
    q = squeeze_env_dim(franka.get_dofs_position())
    dq = squeeze_env_dim(franka.get_dofs_velocity())
    tau = squeeze_env_dim(franka.get_dofs_force())
    print("n_dofs:", franka.n_dofs)
    print("q   (joint position):", q)
    print("dq  (joint velocity):", dq)
    print("tau (joint force)   :", tau)

    #### 2.6 DOF parameters: gains, damping, armature, inverse weight ####
    # APIs:
    #   franka.get_dofs_kp()        -> position gains (Kp)
    #   franka.get_dofs_kv()        -> velocity gains (Kv)
    #   franka.get_dofs_armature()  -> rotor inertia / armature
    #   franka.get_dofs_damping()   -> joint damping
    #   franka.get_dofs_invweight() -> "inverse mass-like" term used internally
    #
    # Use when:
    #   - You want to tune PD controllers for position/velocity control.
    #   - You want to understand the effective inertia/damping of each joint.
    print("\n================ DOF inertial / damping parameters ================")
    kp = squeeze_env_dim(franka.get_dofs_kp())
    kv = squeeze_env_dim(franka.get_dofs_kv())
    arm = squeeze_env_dim(franka.get_dofs_armature())
    damp = squeeze_env_dim(franka.get_dofs_damping())
    invw = squeeze_env_dim(franka.get_dofs_invweight())
    print("kp        (position gain):", kp)
    print("kv        (velocity gain):", kv)
    print("armature  (rotor inertia):", arm)
    print("damping   :", damp)
    print("invweight (inverse mass-like term):", invw)

    #### 2.7 Joint-space mass matrix M(q) ####
    # API: franka.get_mass_mat()
    #
    # Returns:
    #   - Single env: (n_dofs, n_dofs) tensor
    #   - Multi env:  (n_envs, n_dofs, n_dofs)
    #
    # Meaning:
    #   M(q) is the standard joint-space inertia matrix in robotics:
    #       M(q) * ddq + C(q,dq) + g(q) = tau
    #
    # Use when:
    #   - Implementing inverse dynamics / operational space control
    #   - Computing kinetic energy: 0.5 * dq^T * M * dq
    #   - Doing dynamically-consistent metrics or sampling
    print("\n================ Mass matrix M(q) ================")
    M = squeeze_env_dim(franka.get_mass_mat())
    print("M shape:", tuple(M.shape))
    print(M)


#### 3. SCRIPT ENTRY: build scene, add Franka, call the "API demo" ####
# Steps:
#   1) Initialize Genesis (choose CPU or GPU backend).
#   2) Create a Scene.
#   3) Add a plane and the Franka robot MJCF.
#   4) Build the scene (VERY IMPORTANT before using get_* APIs).
#   5) Step a few times.
#   6) Call print_franka_dynamics(franka) to view all the info.
def main():
    #### 3.1 Initialize Genesis backend ####
    # Options:
    #   gs.cpu  : run on CPU
    #   gs.gpu  : run on GPU (if you have a supported GPU)
    gs.init(backend=gs.cpu)

    #### 3.2 Create a scene ####
    # show_viewer:
    #   - False: no GUI window (good for servers / headless)
    #   - True : open a viewer window so you can see the robot
    scene = gs.Scene(show_viewer=False)

    #### 3.3 Add a ground plane ####
    # Plane has no DOFs; it's just a static ground object.
    plane = scene.add_entity(gs.morphs.Plane())

    #### 3.4 Add the Franka robot from MJCF file ####
    # file path:
    #   Change "xml/franka_emika_panda/panda.xml" to your own path if needed.
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )

    #### 3.5 Build the scene ####
    # This compiles the simulation and prepares all internal data structures.
    # You MUST call scene.build() before calling any get_* methods.
    scene.build()

    #### 3.6 Step the simulation a few times ####
    # This lets the robot settle and ensures states (q, dq, etc.) are updated.
    for _ in range(10):
        scene.step()

    #### 3.7 Call our "API demo" function ####
    # This will print all the dynamics-related information we described above.
    print_franka_dynamics(franka)


#### 4. STANDARD PYTHON ENTRY POINT ####
if __name__ == "__main__":
    main()
