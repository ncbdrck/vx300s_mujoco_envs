#!/usr/bin/env python3
"""
Train an SB3 SAC+HER policy on the goal-conditioned VX300S MuJoCo reach task.

Standard env id: ``VX300SMujocoReacherGoalSim-v0``

Default (self-launch): one command brings up roscore + the MuJoCo server + controllers and trains:
    rosrun vx300s_mujoco_envs vx300s_mujoco_reach_goal_train.py
Pass --attach to instead connect to a stack started with the package launch file.

The env exposes a Dict observation (observation/achieved_goal/desired_goal); training uses the
standard SAC with use_her=True (Hindsight Experience Replay). Hyper-parameters live in this
package's config/vx300s_reacher_goal_sac.yaml. Models and logs are written under this package.
"""
from __future__ import annotations

import argparse
import sys

import uniros as gym  # subprocess-isolated env proxy that wraps gym.Env (see UniROS docs)

# Trigger env registration.
from vx300s_mujoco_envs.task_envs.reach import vx300s_mujoco_reach_goal  # noqa: F401

from sb3_ros_support.sac import SAC

# Goal (HER) envs expose a Dict observation; the action wrapper and time-limit wrapper apply.
from multiros.wrappers.normalize_action_wrapper import NormalizeActionWrapper
from multiros.wrappers.time_limit_wrapper import TimeLimitWrapper


ENV_ID = "VX300SMujocoReacherGoalSim-v0"

CONFIG_PKG = "vx300s_mujoco_envs"
CONFIG_FILE = "vx300s_reacher_goal_sac.yaml"

PKG = "vx300s_mujoco_envs"
SAVE_PATH = "/models/sim/sac/vx300s/reach_goal/"
LOG_PATH = "/logs/sim/sac/vx300s/reach_goal/"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--attach", action="store_true",
                   help="Attach to a simulation already started with the package launch file "
                        "instead of letting the env launch its own MuJoCo server + roscore.")
    p.add_argument("--no-realtime", action="store_true",
                   help="Use the paused MDP loop instead of the real-time loop.")
    p.add_argument("--fast", action="store_true",
                   help="Deterministic-step mode: advance the sim with the MuJoCo step action "
                        "(no wall-clock sleep) so training runs as fast as the CPU allows. Implies "
                        "non-real-time. Use --fast-steps to set how many physics ticks per env step.")
    p.add_argument("--fast-steps", type=int, default=50,
                   help="Physics steps advanced per env step in --fast mode.")
    p.add_argument("--mujoco-gui", action="store_true",
                   help="Show the MuJoCo viewer (only when self-launching, i.e. without --attach).")
    p.add_argument("--steps", type=int, default=None,
                   help="Override the training step count from the YAML config.")
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--max-episode-steps", type=int, default=100)
    p.add_argument("--environment-loop-rate", type=float, default=10.0)
    p.add_argument("--action-cycle-time", type=float, default=0.5)
    p.add_argument("--reward-type", default="Sparse")
    return p.parse_args()


def build_env(args: argparse.Namespace):
    if args.fast:
        realtime_mode = False
        sim_step_mode = 2
        num_mujoco_steps = args.fast_steps
        action_cycle_time = 0.0
    else:
        realtime_mode = not args.no_realtime
        sim_step_mode = 1
        num_mujoco_steps = 1
        action_cycle_time = args.action_cycle_time

    env_kwargs = dict(
        seed=args.seed,
        realtime_mode=realtime_mode,
        sim_step_mode=sim_step_mode,
        num_mujoco_steps=num_mujoco_steps,
        delta_action=True,
        environment_loop_rate=args.environment_loop_rate,
        action_cycle_time=action_cycle_time,
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
    env.reset()

    model = SAC(env, SAVE_PATH, LOG_PATH, model_pkg_path=PKG,
                config_file_pkg=CONFIG_PKG, config_filename=CONFIG_FILE,
                use_her=True, seed=args.seed)
    # Optional CLI override of the YAML training_steps (e.g. a short smoke run).
    if args.steps is not None:
        model.parm_dict["training_steps"] = args.steps
    model.train()  # BasicModel.train() already saves the trained model
    model.close_env()
    return 0


if __name__ == "__main__":
    sys.exit(main())
