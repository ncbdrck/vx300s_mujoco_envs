#!/usr/bin/env python3
"""
Validate a trained SAC policy on the VX300S MuJoCo reach task.

Mirrors the train script for env construction; loads a saved model and rolls it out
deterministically for ``--episodes`` episodes, reporting the success rate.

Default (self-launch): one command brings up the sim + controllers and validates:
    rosrun vx300s_mujoco_envs vx300s_mujoco_reach_validate.py
Pass --attach to instead connect to a stack started with the package launch file.
"""
from __future__ import annotations

import argparse
import sys

import rospy
import uniros as gym  # subprocess-isolated env proxy that wraps gym.Env (see UniROS docs)

# Trigger env registration.
from vx300s_mujoco_envs.task_envs.reach import vx300s_mujoco_reach  # noqa: F401

from sb3_ros_support.sac import SAC

from multiros.wrappers.normalize_action_wrapper import NormalizeActionWrapper
from multiros.wrappers.normalize_obs_wrapper import NormalizeObservationWrapper
from multiros.wrappers.time_limit_wrapper import TimeLimitWrapper


ENV_ID = "VX300SMujocoReacherSim-v0"

CONFIG_PKG = "vx300s_mujoco_envs"
CONFIG_FILE = "vx300s_reacher_sac.yaml"

# Path is relative to this package (model_pkg below).
PKG = "vx300s_mujoco_envs"
MODEL_BASE = "/models/sim/sac/vx300s/reach/"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--attach", action="store_true",
                   help="Attach to a simulation already started with the package launch file "
                        "instead of letting the env launch its own MuJoCo server + roscore. "
                        "By default the env self-launches everything (one-command workflow).")
    p.add_argument("--no-realtime", action="store_true",
                   help="Use the paused MDP loop instead of the real-time loop.")
    p.add_argument("--mujoco-gui", action="store_true",
                   help="Show the MuJoCo viewer (only when self-launching, i.e. without --attach).")
    p.add_argument("--eval-seed", type=int, default=1000,
                   help="RNG seed for the evaluation env (held-out goal stream).")
    p.add_argument("--max-episode-steps", type=int, default=100)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--model-tag", default="trained_model_reach")
    p.add_argument("--reward-type", default="Dense")
    return p.parse_args()


def build_env(args: argparse.Namespace):
    env_kwargs = dict(
        seed=args.eval_seed,
        realtime_mode=not args.no_realtime,
        delta_action=True,
        environment_loop_rate=10.0,
        action_cycle_time=0.5,
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
    env = NormalizeObservationWrapper(env)
    env = TimeLimitWrapper(env, max_episode_steps=args.max_episode_steps)
    return env


def main() -> int:
    args = parse_args()
    env = build_env(args)

    model_path = MODEL_BASE + args.model_tag
    # The model and the SAC config both live in this package.
    model = SAC.load_trained_model(model_path=model_path, model_pkg=PKG,
                                   config_filename=CONFIG_FILE, config_file_pkg=CONFIG_PKG,
                                   env=env)

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
