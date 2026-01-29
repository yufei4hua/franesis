import genesis as gs

gs.init(backend=gs.cpu)

scene = gs.Scene(show_viewer=True)

plane = scene.add_entity(
    gs.morphs.Plane(),
)
franka = scene.add_entity(
    gs.morphs.MJCF(file="franesis/envs/franka_emika_panda/panda_cylinder.xml"),
)

scene.build()
for i in range(1000):
    scene.step()