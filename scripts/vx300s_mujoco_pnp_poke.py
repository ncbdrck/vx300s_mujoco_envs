#!/usr/bin/env python3
"""
Deterministic grasp + lift check for the VX300S MuJoCo pick-and-place task.

This bypasses the policy to answer one question: can the gripper actually grasp the cube and
lift it? It homes the arm (gripper open), places the cube at a known spot, then runs a scripted
pick-and-place: approach above the cube, descend, close the gripper, lift, and (optionally) move
to a place location. It prints the cube height before/after to confirm a successful grasp.

If the cube rises with the gripper, contact + grasp force + the finger controller all work. If it
stays on the table, the gripper gains likely need tuning (see vx300s_mujoco_control.yaml,
gripper_controller) or the descent is being rejected by the safety floor (watch [SAFETY] warnings).

Run (attach mode, recommended so you can watch in the viewer):
    1. roslaunch vx300s_mujoco_envs vx300s_mujoco_pnp.launch gui:=true
    2. rosrun vx300s_mujoco_envs vx300s_mujoco_pnp_poke.py --attach

Or let it self-launch:
    rosrun vx300s_mujoco_envs vx300s_mujoco_pnp_poke.py --mujoco-gui
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import rospy

# Trigger env registration.
from vx300s_mujoco_envs.task_envs.pnp import vx300s_mujoco_pnp  # noqa: F401
import gymnasium as gym

from multiros.utils import mujoco_models


ENV_ID = "VX300SMujocoPnpSim-v0"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--attach", action="store_true",
                   help="Attach to a sim already started with the package launch file.")
    p.add_argument("--mujoco-gui", action="store_true",
                   help="Show the MuJoCo viewer when self-launching.")
    p.add_argument("--cube-x", type=float, default=0.33, help="Cube x to place it at (m).")
    p.add_argument("--cube-y", type=float, default=0.00, help="Cube y to place it at (m).")
    p.add_argument("--cube-z", type=float, default=0.02, help="Cube z (centre, on the table).")
    p.add_argument("--grasp-z", type=float, default=0.04,
                   help="EE height when closing on the cube (m). ~ cube centre height.")
    p.add_argument("--lift-z", type=float, default=0.20, help="Height to lift the cube to (m).")
    p.add_argument("--settle", type=float, default=1.5, help="Seconds to wait after each move.")
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
    ok, q = env.calculate_ik(target_pos=np.asarray(xyz, dtype=np.float32))
    if not ok:
        rospy.logwarn(f"IK failed for target {xyz}")
        return False
    env.move_arm_joints(q_positions=q, time_from_start=max(0.5, settle * 0.8))
    rospy.sleep(settle)
    return True


def main() -> int:
    args = parse_args()
    env = build_env(args)
    env.reset()

    # Place the cube at a known location.
    mujoco_models.mujoco_set_body_state(body_name=env.cube_body_name,
                                        pos_x=args.cube_x, pos_y=args.cube_y, pos_z=args.cube_z,
                                        set_pose=True, set_twist=True, server_name=env.server_name)
    rospy.sleep(0.5)

    start = env.get_cube_pose()
    rospy.loginfo(f"[GRASP] cube start: {np.round(start, 4)} (z={start[2]:.4f})")

    # Open gripper, approach above the cube.
    env._set_gripper(env.gripper_max)
    move_ee_to(env, [args.cube_x, args.cube_y, 0.18], args.settle)
    # Descend to grasp height.
    move_ee_to(env, [args.cube_x, args.cube_y, args.grasp_z], args.settle)
    # Close the gripper on the cube.
    rospy.loginfo("[GRASP] closing gripper")
    env._set_gripper(env.gripper_min)
    rospy.sleep(args.settle)
    # Lift.
    move_ee_to(env, [args.cube_x, args.cube_y, args.lift_z], args.settle * 1.5)

    lifted = env.get_cube_pose()
    dz = float(lifted[2] - start[2])
    rospy.loginfo(f"[GRASP] cube after lift: {np.round(lifted, 4)} (z={lifted[2]:.4f})")
    rospy.loginfo(f"[GRASP] cube rose {dz:.4f} m")

    if dz > 0.05:
        rospy.loginfo("[GRASP] RESULT: cube lifted — grasp + finger controller work.")
    else:
        rospy.logwarn("[GRASP] RESULT: cube did NOT lift. Check: (a) gripper gains "
                      "(vx300s_mujoco_control.yaml gripper_controller) — effort may be too low to "
                      "hold the cube; (b) safety floor rejecting the descent (see [SAFETY] warnings); "
                      "(c) grasp height / cube friction.")

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
