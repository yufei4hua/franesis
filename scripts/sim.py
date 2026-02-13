"""Simulate Franka specific controller."""

import fire
import genesis as gs  # noqa: F401
import numpy as np

from franesis.control.impedance_controller import CartesianImpedanceController
from franesis.envs.franka_env import FrankaEnv


def main(controller: str = "impedance", n_runs: int = 1, render: bool = True):
    match controller:
        case "impedance":
            controller_cls = CartesianImpedanceController
        case _:
            raise ValueError(f"Unsupported controller: {controller}")

    env = FrankaEnv(render=render)

    for ep in range(n_runs):
        obs, info = env.reset()
        controller = controller_cls(obs=obs, info=info)

        done = np.zeros(1)
        while not done.any():
            action = controller.compute_control(obs=obs, info=info)
            obs, _, done, info = env.step(action)
            controller.step_callback(action, obs, 0.0, done, info)
            print("step:", env.steps[0].item())
            for k, v in obs.items():
                print(f"{k}: {v}")
            print("done:", done)

    controller.episode_callback()


if __name__ == "__main__":
    fire.Fire(main, serialize=lambda _: None)
