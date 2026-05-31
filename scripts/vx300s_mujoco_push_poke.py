#!/usr/bin/env python3
"""
Deterministic contact check for the VX300S MuJoCo push task.

Random joint-delta actions almost never bring the end-effector down to the cube, so they are a
poor test of whether cube contact / sliding physics work. This script bypasses the policy: it
homes the arm, places the cube at a known spot, then drives the end-effector through a scripted
approach-and-push and prints the cube pose before/after. If the cube moves, contact + the
free-joint dynamics + set/get_body_state are all working.

Run (attach mode, recommended so you can watch in the viewer):
    1. roslaunch vx300s_mujoco_envs vx300s_mujoco_push.launch gui:=true
    2. rosrun vx300s_mujoco_envs vx300s_mujoco_push_poke.py --attach

Or let it self-launch:
    rosrun vx300s_mujoco_envs vx300s_mujoco_push_poke.py --mujoco-gui
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import rospy

# Trigger env registration.
from vx300s_mujoco_envs.task_envs.push import vx300s_mujoco_push  # noqa: F401
import gymnasium as gym

from multiros.utils import mujoco_models


ENV_ID = "VX300SMujocoPushSim-v0"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--attach", action="store_true",
                   help="Attach to a sim already started with the package launch file.")
    p.add_argument("--mujoco-gui", action="store_true",
                   help="Show the MuJoCo viewer when self-launching.")
    p.add_argument("--cube-x", type=float, default=0.30, help="Cube x to place it at (m).")
    p.add_argument("--cube-y", type=float, default=0.00, help="Cube y to place it at (m).")
    p.add_argument("--cube-z", type=float, default=0.02, help="Cube z (centre, on the table).")
    p.add_argument("--push-dist", type=float, default=0.10, help="How far in +x to push (m).")
    p.add_argument("--ee-z", type=float, default=0.03,
                   help="EE height for the push (m). Lower = more side-contact with the cube.")
    p.add_argument("--settle", type=float, default=1.0, help="Seconds to wait after each move.")
    return p.parse_args()


def build_env(args):
    kwargs = dict(realtime_mode=True, environment_loop_rate=10.0, action_cycle_time=0.1,
                  random_cube_spawn=False, random_goal=False, log_internal_state=True)
    if args.attach:
        kwargs.update(launch_mujoco=False, new_roscore=False, load_robot=False)
    else:
        kwargs.update(launch_mujoco=True, new_roscore=True, load_robot=True,
                      mujoco_gui=args.mujoco_gui)
    return gym.make(ENV_ID, **kwargs).unwrapped


def move_ee_to(env, xyz, settle):
    """IK to a Cartesian target, command the joints, wait for the motion to settle."""
    ok, q = env.calculate_ik(target_pos=np.asarray(xyz, dtype=np.float32))
    if not ok:
        rospy.logwarn(f"IK failed for target {xyz}")
        return False
    env.move_arm_joints(q_positions=q, time_from_start=max(0.5, settle * 0.8))
    rospy.sleep(settle)
    return True


def cube_xyz(env):
    return env.get_cube_pose()


def main() -> int:
    args = parse_args()
    env = build_env(args)
    env.reset()

    # Place the cube at a known location (no spawn/delete; the body already exists).
    mujoco_models.mujoco_set_body_state(body_name=env.cube_body_name,
                                        pos_x=args.cube_x, pos_y=args.cube_y, pos_z=args.cube_z,
                                        set_pose=True, set_twist=True, server_name=env.server_name)
    rospy.sleep(0.5)

    start = cube_xyz(env)
    rospy.loginfo(f"[POKE] cube start: {np.round(start, 4)}")

    # 1) move above the cube, slightly behind it in -x
    move_ee_to(env, [args.cube_x - 0.08, args.cube_y, 0.15], args.settle)
    # 2) lower to push height behind the cube
    move_ee_to(env, [args.cube_x - 0.08, args.cube_y, args.ee_z], args.settle)
    # 3) push forward through the cube in +x
    move_ee_to(env, [args.cube_x + args.push_dist, args.cube_y, args.ee_z], args.settle * 1.5)

    end = cube_xyz(env)
    moved = float(np.linalg.norm(np.asarray(end)[:2] - np.asarray(start)[:2]))
    rospy.loginfo(f"[POKE] cube end:   {np.round(end, 4)}")
    rospy.loginfo(f"[POKE] cube moved {moved:.4f} m in the table plane")

    if moved > 0.01:
        rospy.loginfo("[POKE] RESULT: cube moved — contact + free-joint physics work.")
    else:
        rospy.logwarn("[POKE] RESULT: cube did NOT move. Check: (a) EE actually reached push "
                      "height (safety floor may have rejected the lower waypoints — see "
                      "[SAFETY] warnings above), (b) cube friction/contact, (c) IK reachability.")

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
