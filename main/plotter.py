import matplotlib.pyplot as plt
import numpy as np
import csv
import argparse
from pathlib import Path


def _try_float(value):
	try:
		return float(value)
	except (TypeError, ValueError):
		return np.nan


def _read_csv_rows(csv_path):
	with open(csv_path, mode="r", newline="") as f:
		reader = csv.DictReader(f)
		rows = list(reader)
	if not rows:
		raise ValueError(f"No rows found in CSV: {csv_path}")
	return rows


def _find_column(fieldnames, candidates):
	lowered = {name.lower(): name for name in fieldnames}
	for candidate in candidates:
		if candidate.lower() in lowered:
			return lowered[candidate.lower()]
	return None


def _get_numeric_column(rows, fieldname):
	return np.array([_try_float(row.get(fieldname)) for row in rows], dtype=np.float64)


def _plot_training_losses(training_csv, outdir, show):
	rows = _read_csv_rows(training_csv)
	fieldnames = rows[0].keys()

	step_col = _find_column(fieldnames, ["step", "steps", "timestep", "timesteps"])
	actor_col = _find_column(fieldnames, ["actor_loss", "actor"])
	critic1_col = _find_column(fieldnames, ["critic_1_loss", "critic1_loss", "critic_loss_1"])
	critic2_col = _find_column(fieldnames, ["critic_2_loss", "critic2_loss", "critic_loss_2"])

	missing = [
		name for name, col in [
			("step", step_col),
			("actor_loss", actor_col),
			("critic_1_loss", critic1_col),
			("critic_2_loss", critic2_col),
		] if col is None
	]
	if missing:
		raise ValueError(
			f"Training CSV is missing required columns: {', '.join(missing)}"
		)

	steps = _get_numeric_column(rows, step_col)
	actor_loss = _get_numeric_column(rows, actor_col)
	critic_1_loss = _get_numeric_column(rows, critic1_col)
	critic_2_loss = _get_numeric_column(rows, critic2_col)

	fig, ax = plt.subplots(figsize=(10, 5))
	ax.plot(steps, actor_loss, label="Actor Loss", linewidth=1.5)
	ax.plot(steps, critic_1_loss, label="Critic 1 Loss", linewidth=1.5)
	ax.plot(steps, critic_2_loss, label="Critic 2 Loss", linewidth=1.5)
	ax.set_title("Training Losses Over Steps")
	ax.set_xlabel("Step")
	ax.set_ylabel("Loss")
	ax.grid(True, alpha=0.3)
	ax.legend()
	fig.tight_layout()
	fig.savefig(outdir / "training_losses_over_steps.png", dpi=150)
	if show:
		plt.show()
	plt.close(fig)


def _plot_episode_rewards(episode_csv, outdir, show):
	rows = _read_csv_rows(episode_csv)
	fieldnames = rows[0].keys()

	episode_col = _find_column(fieldnames, ["episode", "episodes", "ep"])
	reward_col = _find_column(fieldnames, ["reward", "episode_reward"])
	discounted_col = _find_column(fieldnames, ["discounted_reward", "discounted_rewards", "disc_reward"])

	missing = [
		name for name, col in [
			("episode", episode_col),
			("reward", reward_col),
			("discounted_reward", discounted_col),
		] if col is None
	]
	if missing:
		raise ValueError(
			f"Episode CSV is missing required columns: {', '.join(missing)}"
		)

	episodes = _get_numeric_column(rows, episode_col)
	rewards = _get_numeric_column(rows, reward_col)
	discounted_rewards = _get_numeric_column(rows, discounted_col)

	fig, ax = plt.subplots(figsize=(10, 5))
	ax.plot(episodes, rewards, label="Reward", linewidth=1.5)
	ax.plot(episodes, discounted_rewards, label="Discounted Reward", linewidth=1.5)
	ax.set_title("Rewards Over Episodes")
	ax.set_xlabel("Episode")
	ax.set_ylabel("Reward")
	ax.grid(True, alpha=0.3)
	ax.legend()
	fig.tight_layout()
	fig.savefig(outdir / "rewards_and_discounted_over_episodes.png", dpi=150)
	if show:
		plt.show()
	plt.close(fig)


def _plot_episode_distance(episode_csv, outdir, show):
	rows = _read_csv_rows(episode_csv)
	fieldnames = rows[0].keys()

	episode_col = _find_column(fieldnames, ["episode", "episodes", "ep"])
	distance_col = _find_column(fieldnames, ["distance_traveled", "distance", "dist_traveled"])

	missing = [
		name for name, col in [
			("episode", episode_col),
			("distance_traveled", distance_col),
		] if col is None
	]
	if missing:
		raise ValueError(
			f"Episode CSV is missing required columns: {', '.join(missing)}"
		)

	episodes = _get_numeric_column(rows, episode_col)
	distance = _get_numeric_column(rows, distance_col)

	fig, ax = plt.subplots(figsize=(10, 5))
	ax.plot(episodes, distance, label="Distance Traveled", linewidth=1.5)
	ax.set_title("Distance Traveled Over Episodes")
	ax.set_xlabel("Episode")
	ax.set_ylabel("Distance")
	ax.grid(True, alpha=0.3)
	ax.legend()
	fig.tight_layout()
	fig.savefig(outdir / "distance_over_episodes.png", dpi=150)
	if show:
		plt.show()
	plt.close(fig)


def _plot_episode_steps(episode_csv, outdir, show):
	rows = _read_csv_rows(episode_csv)
	fieldnames = rows[0].keys()

	episode_col = _find_column(fieldnames, ["episode", "episodes", "ep"])
	steps_col = _find_column(fieldnames, ["steps", "episode_steps", "ep_steps"])

	missing = [
		name for name, col in [
			("episode", episode_col),
			("steps", steps_col),
		] if col is None
	]
	if missing:
		raise ValueError(
			f"Episode CSV is missing required columns: {', '.join(missing)}"
		)

	episodes = _get_numeric_column(rows, episode_col)
	steps = _get_numeric_column(rows, steps_col)

	fig, ax = plt.subplots(figsize=(10, 5))
	ax.plot(episodes, steps, label="Steps per Episode", linewidth=1.5)
	ax.set_title("Steps Over Episodes")
	ax.set_xlabel("Episode")
	ax.set_ylabel("Steps")
	ax.grid(True, alpha=0.3)
	ax.legend()
	fig.tight_layout()
	fig.savefig(outdir / "steps_over_episodes.png", dpi=150)
	if show:
		plt.show()
	plt.close(fig)


def parse_args():
	parser = argparse.ArgumentParser(
		description="Generate training and episode plots from TD3 CSV logs."
	)
	parser.add_argument(
		"--training-csv",
		type=Path,
		required=True,
		help="Path to training log CSV (must include step, actor_loss, critic_1_loss, critic_2_loss).",
	)
	parser.add_argument(
		"--episode-csv",
		type=Path,
		required=True,
		help="Path to episode log CSV (must include episode, reward, discounted_reward, steps, distance_traveled).",
	)
	parser.add_argument(
		"--outdir",
		type=Path,
		default=Path("plots"),
		help="Directory to save generated plot images.",
	)
	parser.add_argument(
		"--show",
		action="store_true",
		help="Display plots interactively in addition to saving images.",
	)
	return parser.parse_args()


def main():
    args = parse_args()
    args.training_csv = Path("logs") / args.training_csv
    args.episode_csv = Path("logs") / args.episode_csv

    if not args.training_csv.exists():
        raise FileNotFoundError(f"Training CSV not found: {args.training_csv}")
    if not args.episode_csv.exists():
        raise FileNotFoundError(f"Episode CSV not found: {args.episode_csv}")

    args.outdir.mkdir(parents=True, exist_ok=True)

    _plot_training_losses(args.training_csv, args.outdir, args.show)
    _plot_episode_rewards(args.episode_csv, args.outdir, args.show)
    _plot_episode_distance(args.episode_csv, args.outdir, args.show)
    _plot_episode_steps(args.episode_csv, args.outdir, args.show)

    print(f"Saved plots to: {args.outdir.resolve()}")


if __name__ == "__main__":
	main()

# run with:
# python plotter.py --training-csv clamped_td3_training_log.csv --episode-csv .\clamped_td3_episode_log.csv --outdir .\plots_test
