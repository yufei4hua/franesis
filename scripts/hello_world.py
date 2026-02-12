"""Quick test for FrankaEnv with random torque commands."""

import genesis as gs
import numpy as np
import torch

from franesis.envs.franka_env import FrankaEnv


def main():
    env = FrankaEnv()

    _ = env.reset()
    action_dim = env._arm_dof_dim
    done = np.zeros(1)

    while not done.any():
        random_tau = 1 * (2.0 * torch.rand(1, action_dim, device=gs.device) - 1.0)
        obs, _, done, info = env.step(random_tau)
        print("step:", env.steps[0].item())
        for k, v in obs.items():
            print(f"{k}: {v}")
        for k, v in info.items():
            print(f"{k}: {v}")
        print("done:", done)


if __name__ == "__main__":
    main()
