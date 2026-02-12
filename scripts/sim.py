"""Simulate Franka specific controller."""

import genesis as gs  # noqa: F401
import numpy as np

from franesis.control.impedance import CartesianImpedanceController
from franesis.envs.franka_env import FrankaEnv


def main():
    env = FrankaEnv()

    obs, info = env.reset()
    controller = CartesianImpedanceController(obs=obs, info=info)

    done = np.zeros(1)
    while not done.any():
        action = controller.compute_control(obs=obs, info=info)
        obs, _, done, info = env.step(action)
        print("step:", env.steps[0].item())
        for k, v in obs.items():
            print(f"{k}: {v}")
        for k, v in info.items():
            print(f"{k}: {v}")
        print("done:", done)


if __name__ == "__main__":
    main()
