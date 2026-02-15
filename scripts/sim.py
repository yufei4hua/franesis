"""Simulate Franka specific controller."""

import fire
import genesis as gs  # noqa: F401
import numpy as np

from franesis.control import CartesianImpedanceController, HFICController, MotionForceController, PDIMFCController
from franesis.envs.franka_box_env import FrankaBoxEnv
from franesis.envs.franka_env import FrankaEnv
from franesis.envs.franka_sphere_env import FrankaSphereEnv
from franesis.envs.franka_surface_env import FrankaSurfaceEnv


def main(environment: str = "default", controller: str = "imp", n_runs: int = 1, render: bool = True):
    match controller:
        case "imp":
            controller_cls = CartesianImpedanceController
        case "mfc":
            controller_cls = MotionForceController
        case "hfic":
            controller_cls = HFICController
        case "pdimfc":
            controller_cls = PDIMFCController
        case _:
            raise ValueError(f"Unsupported controller: {controller}")

    match environment:
        case "default":
            env_cls = FrankaEnv
        case "box":
            env_cls = FrankaBoxEnv
        case "surface":
            env_cls = FrankaSurfaceEnv
        case "sphere":
            env_cls = FrankaSphereEnv
        case _:
            raise ValueError(f"Unsupported environment: {environment}")

    env = env_cls(render=render)
    for ep in range(n_runs):
        obs, info = env.reset()
        ctrl = controller_cls(obs=obs, info=info, freq=env.freq)

        done = np.zeros(1)
        while not done.any():
            action = ctrl.compute_control(obs=obs, info=info)
            obs, _, done, info = env.step(action)
            ctrl.step_callback(action, obs, 0.0, done, info)

    ctrl.episode_callback(exp_name=f"{environment}_{controller}")


if __name__ == "__main__":
    fire.Fire(main, serialize=lambda _: None)
