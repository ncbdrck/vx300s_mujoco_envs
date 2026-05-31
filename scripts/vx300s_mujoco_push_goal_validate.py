#!/usr/bin/env python3
"""
Validate a trained SAC+HER policy on the goal-conditioned VX300S MuJoCo push task.

Mirrors the train script for env construction; loads a saved model and rolls it out
deterministically for ``--episodes`` episodes, reporting the success rate.

Default (self-launch): one command brings up the sim + controllers and validates:
    rosrun vx300s_mujoco_envs vx300s_mujoco_reach_goal_validate.py
Pass --attach to instead connect to a stack started with the package launch file.
"""
from __future__ import annotations

import argparse
import sys

import rospy
import uniros as gym  # paper section 6.1: subprocess-isolated env proxy; drop-in for gym.Env

# Trigger env registration.
from vx300s_mujoco_envs.task_envs.push import push  # noqa: F401

from sb3_ros_support.sac import SAC

# Goal (HER) envs expose a Dict observation; the action wrapper and time-limit wrapper apply.
from multiros.wrappers.normalize_action_wrapper import NormalizeActionWrapper
from multiros.wrappers.time_limit_wrapper import TimeLimitWrapper


ENV_ID = "VX300SMujocoPushGoalSim-v0"

CONFIG_PKG = "vx300s_mujoco_envs"
CONFIG_FILE = "vx300s_pusher_goal_sac.yaml"

PKG = "vx300s_mujoco_envs"
MODEL_BASE = "/models/sim/sac/vx300s/push_goal/"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--attach", action="store_true",
                   help="Attach to a simulation already started with the package launch file.")
    p.add_argument("--no-realtime", action="store_true",
                   help="Use the paused MDP loop instead of the real-time (paper section 7) loop.")
    p.add_argument("--mujoco-gui", action="store_true",
                   help="Show the MuJoCo viewer (only when self-launching, i.e. without --attach).")
    p.add_argument("--eval-seed", type=int, default=1000)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--max-episode-steps", type=int, default=100)
    p.add_argument("--model-tag", default="trained_model_push_goal",
                   help="Name of the saved model file (without extension) under the model path.")
    p.add_argument("--reward-type", default="Sparse")
    return p.parse_args()


def build_env(args: argparse.Namespace):
    env_kwargs = dict(
        seed=args.eval_seed,
        realtime_mode=not args.no_realtime,
        delta_action=True,
        action_speed=0.100,
        reward_type=args.reward_type,
        log_internal_state=False,
    )
    if args.attach:
        env_kwargs.update(launch_mujoco=False, new_roscore=False, load_robot=False)
    else:
        env_kwargs.update(launch_mujoco=True, new_roscore=True, load_robot=True,
                          mujoco_gui=args.mujoco_gui)

    env = gym.make(ENV_ID, **env_kwargs)
    env = NormalizeActionWrapper(env)
    env = TimeLimitWrapper(env, max_episode_steps=args.max_episode_steps)
    return env


def main() -> int:
    args = parse_args()
    env = build_env(args)

    model_path = MODEL_BASE + args.model_tag
    # The model and the SAC+HER config both live in this package.
    model = SAC.load_trained_model(model_path=model_path, model_pkg=PKG,
                                   config_filename=CONFIG_FILE, config_file_pkg=CONFIG_PKG,
                                   env=env, use_her=True)

    obs, _ = env.reset()
    successes, truncs = 0, 0
    for ep in range(args.episodes):
        done = False
        ep_success = False
        while not done:
            action, _ = model.predict(observation=obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            if info.get("is_success"):
                ep_success = True
            if terminated or truncated:
                done = True
                if truncated and not terminated:
                    truncs += 1
        if ep_success:
            successes += 1
        rospy.loginfo(f"Episode {ep + 1}/{args.episodes} success={ep_success}")
        obs, _ = env.reset()

    print(f"\nResults over {args.episodes} episodes:")
    print(f"  success rate:        {successes}/{args.episodes} = {100 * successes / args.episodes:.1f}%")
    print(f"  truncated (no term): {truncs}")
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
