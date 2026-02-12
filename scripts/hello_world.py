"""Quick test for FrankaEnv with random torque commands."""

import argparse

import genesis as gs
import torch

from franesis.envs.franka_env import FrankaEnv


def main() -> None:
	parser = argparse.ArgumentParser(description="Test FrankaEnv with random force control")
	parser.add_argument("--steps", type=int, default=500, help="Simulation steps")
	parser.add_argument("--num-envs", type=int, default=1, help="Number of vectorized environments")
	args = parser.parse_args()

	# Initialize Genesis backend.
	gs.init(backend=gs.gpu)

	env = FrankaEnv(
		show_viewer=True,
		num_envs=args.num_envs,
		render=False,
	)

	_ = env.reset()
	action_dim = env._arm_dof_dim

	for i in range(args.steps):
		random_tau = 10 * (2.0 * torch.rand(args.num_envs, action_dim, device=gs.device) - 1.0)
		obs, _, done, _ = env.step(random_tau)

		if i % 50 == 0:
			q = obs["q"][0]
			ee = obs["ee_pos"][0]
			print(
				f"step={i:04d} | q0={q[0].item(): .3f} | "
				f"ee=({ee[0].item(): .3f}, {ee[1].item(): .3f}, {ee[2].item(): .3f}) | "
				f"done_any={bool(done.any().item())}"
			)


if __name__ == "__main__":
	main()
