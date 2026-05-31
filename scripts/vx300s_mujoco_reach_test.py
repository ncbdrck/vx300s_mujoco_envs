#!/usr/bin/env python3
"""
Smoke-test the VX300S MuJoCo reach environment with random actions.

Default (self-launch): one command brings up roscore + the MuJoCo server + controllers and runs
the smoke test, mirroring the one-call workflow of the rl_environments envs:
    rosrun vx300s_mujoco_envs vx300s_mujoco_reach_test.py

To instead attach to a stack started separately (useful while debugging the bring-up), set
ATTACH_MODE = True below and first run:
    roslaunch vx300s_mujoco_envs vx300s_mujoco_reach.launch gui:=true

The environment creates its own ROS node during construction, so this script does not call
rospy.init_node itself.
"""

import sys

import gymnasium as gym
from gymnasium.wrappers import TimeLimit

# Environment registration (importing the module registers "VX300SMujocoReacherSim-v0").
from vx300s_mujoco_envs.task_envs.reach import vx300s_mujoco_reach  # noqa: F401

# wrappers
from multiros.wrappers.normalize_action_wrapper import NormalizeActionWrapper


# --- Configuration -----------------------------------------------------------
ENV_ID = 'VX300SMujocoReacherSim-v0'

# Self-launch (default): gym.make brings up roscore + the MuJoCo server + controllers itself,
# so a single command runs everything (same one-call workflow as the rl_environments envs).
# Set ATTACH_MODE = True to instead attach to a stack already started with the package launch
# file (useful while debugging the simulation bring-up).
ATTACH_MODE = False

# Real-time (paper section 7) loop vs. paused MDP loop.
REALTIME_MODE = True

# Loop timing. action_cycle_time must be >= 1 / environment_loop_rate.
ENVIRONMENT_LOOP_RATE = 10.0
ACTION_CYCLE_TIME = 0.1

MAX_EPISODE_STEPS = 100
EPISODES = 1000
SEED = 10


def build_env():
    if ATTACH_MODE:
        env = gym.make(ENV_ID, launch_mujoco=False, new_roscore=False, load_robot=False,
                       realtime_mode=REALTIME_MODE, environment_loop_rate=ENVIRONMENT_LOOP_RATE,
                       action_cycle_time=ACTION_CYCLE_TIME, seed=SEED)
    else:
        env = gym.make(ENV_ID, launch_mujoco=True, new_roscore=True, load_robot=True,
                       mujoco_gui=True, realtime_mode=REALTIME_MODE,
                       environment_loop_rate=ENVIRONMENT_LOOP_RATE,
                       action_cycle_time=ACTION_CYCLE_TIME, seed=SEED)

    env = NormalizeActionWrapper(env)
    env = TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
    return env


def main():
    env = build_env()

    obs, _ = env.reset()
    epi_count = 0
    while epi_count < EPISODES:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            epi_count += 1
            obs, _ = env.reset()

    env.close()
    sys.exit()


if __name__ == '__main__':
    main()
