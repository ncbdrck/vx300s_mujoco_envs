#!/usr/bin/env python3

from typing import Any, Optional, Dict

import rospy
import rospkg
import numpy as np
from gymnasium import spaces
from urdf_parser_py.urdf import URDF
from gymnasium.envs.registration import register
import scipy.spatial

# Custom robot-goal env
from vx300s_mujoco_envs.robot_envs import vx300s_mujoco_robot_goal

# core modules of the framework
from multiros.utils import mujoco_core
from multiros.utils import ros_common
from multiros.utils import ros_markers


register(
    id='VX300SMujocoReacherGoalSim-v0',
    entry_point='vx300s_mujoco_envs.task_envs.reach.vx300s_mujoco_reach_goal:VX300SMujocoReacherGoalEnv',
    max_episode_steps=100,
)


class VX300SMujocoReacherGoalEnv(vx300s_mujoco_robot_goal.VX300SMujocoRobotGoalEnv):
    """
    Goal-conditioned (HER-ready) reach task for the VX300S robot on the MuJoCo backend.

    Same task as the standard reach env, but exposes the gymnasium-robotics GoalEnv contract: the
    observation is a dict with ``observation`` / ``achieved_goal`` / ``desired_goal``, and the
    reward is computed from (achieved_goal, desired_goal) so Hindsight Experience Replay can relabel
    goals. The achieved goal is the end-effector position; the desired goal is the sampled reach
    target.

    Action Space - Continuous (6 arm-joint deltas).
    Observation  - Dict:
        observation   : EE pos, joint values, previous action, joint velocities (proprioceptive;
                        no goal-derived features, so HER can relabel cleanly).
        achieved_goal : EE position (3,).
        desired_goal  : reach target (3,).
    """

    def __init__(self, launch_mujoco: bool = True, new_roscore: bool = True, roscore_port: str = None,
                 mujoco_paused: bool = False, mujoco_gui: bool = False, model_path: str = None,
                 model_pkg: str = "vx300s_mujoco_envs", model_name: str = "assets/vx300s_mjcf/vx300s_table_scene.xml",
                 server_name: str = "mujoco_server", seed: int = None, reward_type: str = "Sparse",
                 delta_action: bool = True, delta_coeff: float = 0.05,
                 environment_loop_rate: float = 10, action_cycle_time: float = 0.100,
                 action_speed: float = 0.5,
                 log_internal_state: bool = False, debug: bool = False,
                 realtime_mode: bool = True, load_robot: bool = True,
                 sim_step_mode: int = 1, num_mujoco_steps: int = 1):

        self.realtime_mode = realtime_mode

        # Goal (HER) envs always use the sparse reward in compute_reward: HER relabels
        # transitions, so the reward must be a deterministic function of
        # (achieved_goal, desired_goal). reward_type is accepted for signature symmetry
        # with the standard envs but does not select a dense reward here.
        if str(reward_type).lower() != "sparse":
            rospy.logwarn("Goal-conditioned env uses sparse HER rewards; "
                          "ignoring reward_type=%s", reward_type)

        ros_port = None
        mujoco_pid = None

        _pkg_path = rospkg.RosPack().get_path("vx300s_mujoco_envs")
        plugin_config = _pkg_path + "/config/vx300s_mujoco_plugins.yaml"
        initial_joint_states = _pkg_path + "/config/initial_joint_states.yaml"

        if launch_mujoco:
            ros_port, mujoco_pid = self._launch_mujoco(launch_roscore=new_roscore, port=roscore_port,
                                                       paused=mujoco_paused, headless=not mujoco_gui,
                                                       model_path=model_path, model_pkg=model_pkg,
                                                       model_name=model_name, server_name=server_name,
                                                       mujoco_plugin_config=plugin_config,
                                                       initial_joint_states=initial_joint_states)
        elif new_roscore:
            ros_port = self._launch_roscore(port=roscore_port)
        elif roscore_port is not None:
            ros_port = roscore_port
            ros_common.change_ros_master(ros_port)
        else:
            if ros_common.is_roscore_running() is False:
                print("roscore is not running! Launching a new roscore and the MuJoCo server!")
                ros_port, mujoco_pid = mujoco_core.launch_mujoco(launch_roscore=new_roscore,
                                                                 port=roscore_port,
                                                                 paused=mujoco_paused,
                                                                 headless=not mujoco_gui,
                                                                 model_path=model_path,
                                                                 model_pkg=model_pkg,
                                                                 model_name=model_name,
                                                                 server_name=server_name,
                                                                 mujoco_plugin_config=plugin_config,
                                                                 initial_joint_states=initial_joint_states)

        if ros_port is not None:
            self.node_name = "VX300SMujocoReacherGoalEnvSim" + "_" + ros_port
        else:
            self.node_name = "VX300SMujocoReacherGoalEnvSim"

        rospy.init_node(self.node_name, anonymous=True)
        rospy.loginfo(f"Starting {self.node_name}")

        if self.realtime_mode and action_cycle_time > 0.0 \
                and (1.0 / environment_loop_rate) > action_cycle_time:
            rospy.logerr("The environment loop rate is greater than the action cycle time. Exiting the program!")
            rospy.signal_shutdown("Exiting the program!")
            exit()

        self.log_internal_state = log_internal_state
        self.delta_action = delta_action
        self.delta_coeff = delta_coeff
        self.action_cycle_time = action_cycle_time
        self.action_speed = action_speed
        self.debug = debug

        # load task parameters onto the parameter server (shared with the standard reach task)
        ros_common.ros_load_yaml(pkg_name="vx300s_mujoco_envs", file_name="vx300s_reach_task_config.yaml", ns="/")
        self._get_params()

        # Joint action space (6 arm joints)
        self.action_space = spaces.Box(low=np.array(self.min_joint_values), high=np.array(self.max_joint_values),
                                       dtype=np.float32)

        # ---- proprioceptive observation bounds (no goal-derived features)
        observations_high_ee_pos = np.array([self.position_ee_max["x"], self.position_ee_max["y"],
                                             self.position_ee_max["z"]])
        observations_low_ee_pos = np.array([self.position_ee_min["x"], self.position_ee_min["y"],
                                            self.position_ee_min["z"]])
        observations_high_joint_values = self.max_joint_angles.copy()
        observations_low_joint_values = self.min_joint_angles.copy()
        observations_high_prev_action = self.max_joint_values.copy()
        observations_low_prev_action = self.min_joint_values.copy()
        observations_high_joint_vel = self.max_joint_vel.copy()
        observations_low_joint_vel = self.min_joint_vel.copy()

        obs_high = np.concatenate(
            [observations_high_ee_pos, observations_high_joint_values,
             observations_high_prev_action, observations_high_joint_vel, ])
        obs_low = np.concatenate(
            [observations_low_ee_pos, observations_low_joint_values,
             observations_low_prev_action, observations_low_joint_vel, ])

        # ---- goal spaces (achieved = EE pos, desired = reach target); both within the goal region
        goal_high = np.array([self.position_goal_max["x"], self.position_goal_max["y"],
                              self.position_goal_max["z"]])
        goal_low = np.array([self.position_goal_min["x"], self.position_goal_min["y"],
                             self.position_goal_min["z"]])
        # achieved goal (EE pos) can range over the workspace, not just the goal region
        ach_high = np.array([self.position_ee_max["x"], self.position_ee_max["y"], self.position_ee_max["z"]])
        ach_low = np.array([self.position_ee_min["x"], self.position_ee_min["y"], self.position_ee_min["z"]])

        self.observation_space = spaces.Dict(dict(
            observation=spaces.Box(low=obs_low, high=obs_high, dtype=np.float32),
            achieved_goal=spaces.Box(low=ach_low, high=ach_high, dtype=np.float32),
            desired_goal=spaces.Box(low=goal_low, high=goal_high, dtype=np.float32),
        ))

        # ---- goal sampling space
        self.goal_space = spaces.Box(low=goal_low, high=goal_high, dtype=np.float32, seed=seed)

        # ---- workspace (for action validity)
        high_workspace_range = np.array([self.workspace_max["x"], self.workspace_max["y"], self.workspace_max["z"]])
        low_workspace_range = np.array([self.workspace_min["x"], self.workspace_min["y"], self.workspace_min["z"]])
        self.workspace_space = spaces.Box(low=low_workspace_range, high=high_workspace_range, dtype=np.float32)

        self.goal_marker = ros_markers.RosMarker(frame_id="world", ns="goal", marker_type=2, marker_topic="goal_pos",
                                                 lifetime=20.0)

        super().__init__(ros_port=ros_port, mujoco_pid=mujoco_pid, server_name=server_name, seed=seed,
                         real_time=self.realtime_mode, action_cycle_time=action_cycle_time,
                         load_robot=load_robot, sim_step_mode=sim_step_mode,
                         num_mujoco_steps=num_mujoco_steps)

        self.environment_loop_time = 1.0 / environment_loop_rate
        self.prev_action = None
        self.reach_goal = np.zeros(3, dtype=np.float32)

        if environment_loop_rate is not None:
            self.obs_r = None
            self.achieved_r = None
            self.desired_r = None
            self.current_action = None
            self.init_done = False

            if self.debug:
                self.loop_counter = 0
                self.action_counter = 0

            if self.realtime_mode:
                rospy.Timer(rospy.Duration(1.0 / environment_loop_rate), self.environment_loop)

        self.action_not_in_limits = False
        self.movement_result = False
        self.within_goal_space = False

        rospy.loginfo(f"Finished Init of {self.node_name}")

    # -------------------------------------------------------
    #   Goal-env hooks

    def _set_init_params(self, options: Optional[Dict[str, Any]] = None):
        if self.log_internal_state:
            rospy.loginfo("Initialising the init params!")

        self.init_pos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.init_done = False
        self.current_action = None

        self.movement_result = self.move_arm_joints(self.init_pos, time_from_start=self.action_speed,
                                                    await_convergence=True)
        if not self.movement_result and self.log_internal_state:
            rospy.logwarn("Homing failed!")

        self.reach_goal = self._sample_box(self.goal_space)
        self.goal_marker.set_position(position=self.reach_goal)
        self.goal_marker.publish()

        self.ee_pos = np.asarray(self.get_ee_pose(), dtype=np.float32)
        self.joint_values = self.get_joint_angles()

        self.action_not_in_limits = False
        self.within_goal_space = True
        self.prev_action = self.init_pos.copy()

        self.obs_r = None
        self.achieved_r = None
        self.desired_r = None

        if self.debug:
            self.loop_counter = 0
            self.action_counter = 0

        self.init_done = True

    def _set_action(self, action):
        self.prev_action = np.asarray(action, dtype=np.float32).copy()
        self.current_action = np.asarray(action, dtype=np.float32).copy()
        if self.debug:
            self.action_counter = 0
        if not self.realtime_mode:
            self.obs_r = None
            self.execute_action(action)

    def _get_observation(self):
        if self.obs_r is not None:
            return self.obs_r.copy()
        return self.sample_observation()

    def _get_achieved_goal(self):
        if self.achieved_r is not None:
            return self.achieved_r.copy()
        return np.asarray(self.ee_pos, dtype=np.float32)

    def _get_desired_goal(self):
        if self.desired_r is not None:
            return self.desired_r.copy()
        return np.asarray(self.reach_goal, dtype=np.float32)

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Sparse HER reward: 0 when the achieved goal is within the reach tolerance of the desired
        goal, else -1. Batch-safe: SB3's HER calls this with (N, 3) arrays as well as single (3,)
        goals, so the distance is computed along the last axis and the result keeps that shape.
        """
        achieved = np.asarray(achieved_goal, dtype=np.float32)
        desired = np.asarray(desired_goal, dtype=np.float32)
        distance = np.linalg.norm(achieved - desired, axis=-1)
        return -(distance > self.reach_tolerance).astype(np.float32)

    def compute_terminated(self, achieved_goal, desired_goal, info):
        distance = np.linalg.norm(np.asarray(achieved_goal) - np.asarray(desired_goal), axis=-1)
        # Batch-safe like compute_reward: scalar in -> bool, (N,) in -> bool array.
        terminated = distance <= self.reach_tolerance
        return bool(terminated) if np.ndim(terminated) == 0 else terminated

    def compute_truncated(self, achieved_goal, desired_goal, info):
        return False

    # -------------------------------------------------------
    #   Real-time loop + action execution

    def environment_loop(self, event):
        if self.init_done:
            if rospy.is_shutdown():
                return
            jv = getattr(self, "joint_values", None)
            if jv is None or len(jv) < 6:
                return
            if self.debug:
                self.loop_counter += 1
            self.obs_r = self.sample_observation()
            self.achieved_r = np.asarray(self.ee_pos, dtype=np.float32)
            self.desired_r = np.asarray(self.reach_goal, dtype=np.float32)
            if self.current_action is not None:
                self.execute_action(self.current_action)
                if self.debug:
                    self.action_counter += 1

    def execute_action(self, action):
        if self.delta_action:
            self.joint_values = self.get_joint_angles()
            if self.joint_values is None or len(self.joint_values) < len(self.min_joint_values):
                if self.log_internal_state:
                    rospy.logwarn("Joint action rejected: current joint vector is stale or empty.")
                self.movement_result = False
                self.within_goal_space = False
                return
            action = np.asarray(self.joint_values) + (np.asarray(action) * self.delta_coeff)

        action = np.clip(action, self.min_joint_values, self.max_joint_values)

        if self.check_action_within_workspace(action):
            safe, reason = self._check_action_links_safe(action, current_joints=self.joint_values)
            if not safe:
                if self.log_internal_state:
                    rospy.logwarn(f"[SAFETY] joint action rejected: {reason}")
                self.movement_result = False
                self.within_goal_space = False
            else:
                self.movement_result = self.move_arm_joints(q_positions=action, time_from_start=self.action_speed)
                self.within_goal_space = True
        else:
            self.movement_result = False
            self.within_goal_space = False

    def sample_observation(self):
        ee = self.fk_pykdl(self.get_joint_angles())
        if ee is None:
            ee = self.ee_pos
        self.ee_pos = np.asarray(ee, dtype=np.float32)
        self.joint_values = list(self.joint_pos_all[:len(self.arm_joint_names)])

        if self.prev_action is None:
            prev_action = self.get_joint_angles()
        else:
            prev_action = self.prev_action.copy()

        obs = np.concatenate((self.ee_pos, self.joint_pos_all[:len(self.arm_joint_names)],
                              prev_action, self.current_joint_velocities[:len(self.arm_joint_names)]),
                             axis=None, dtype=np.float32)
        return obs.copy()

    def check_action_within_workspace(self, action):
        ee_pos = self.fk_pykdl(action=action)
        if ee_pos is not None:
            return bool(self.workspace_space.contains(ee_pos))
        return False

    # Controlled arm joints, in mujoco_ros_control registration / joint_states order.
    ARM_JOINT_NAMES = ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")

    def _arm_joint_limits_from_urdf(self):
        urdf_pkg_name = "viperx300s_description"
        urdf_file_name = "vx300s.urdf.xacro"
        _, urdf_string = ros_common.load_urdf(pkg_name=urdf_pkg_name,
                                              file_name=urdf_file_name,
                                              folder="/urdf", param_name=None)
        robot = URDF.from_xml_string(urdf_string)
        by_name = {j.name: j for j in robot.joints}
        lower, upper, vel = [], [], []
        for name in self.ARM_JOINT_NAMES:
            j = by_name.get(name)
            if j is None or j.limit is None:
                raise RuntimeError(
                    f"Arm joint '{name}' or its <limit> missing from "
                    f"{urdf_pkg_name}/urdf/{urdf_file_name}")
            lower.append(float(j.limit.lower))
            upper.append(float(j.limit.upper))
            vel.append(float(j.limit.velocity))
        return lower, upper, vel

    def _get_params(self):
        self.min_joint_values = rospy.get_param('/vx300s/min_joint_pos')
        self.max_joint_values = rospy.get_param('/vx300s/max_joint_pos')

        self.position_ee_max = rospy.get_param('/vx300s/position_ee_max')
        self.position_ee_min = rospy.get_param('/vx300s/position_ee_min')

        lower, upper, vel = self._arm_joint_limits_from_urdf()
        self.min_joint_angles = lower
        self.max_joint_angles = upper
        self.min_joint_vel = [-v for v in vel]
        self.max_joint_vel = list(vel)

        self.position_goal_max = rospy.get_param('/vx300s/position_goal_max')
        self.position_goal_min = rospy.get_param('/vx300s/position_goal_min')

        self.reach_tolerance = rospy.get_param('/vx300s/reach_tolerance')

        self.workspace_max = rospy.get_param('/vx300s/workspace_max')
        self.workspace_min = rospy.get_param('/vx300s/workspace_min')

    # -------------------------------------------------------
    #   Launch helpers

    def _launch_mujoco(self, launch_roscore=True, port=None, paused=False, use_sim_time=True,
                       model_path=None, model_pkg=None, model_name=None, headless=True,
                       no_render=False, realtime=None, mujoco_plugin_config=None,
                       initial_joint_states=None, server_name="mujoco_server", ns="",
                       verbose=False, output='screen', launch_new_term=True):
        ros_port, mujoco_pid = mujoco_core.launch_mujoco(
            launch_roscore=launch_roscore, port=port, paused=paused, use_sim_time=use_sim_time,
            model_path=model_path, model_pkg=model_pkg, model_name=model_name, headless=headless,
            no_render=no_render, realtime=realtime, mujoco_plugin_config=mujoco_plugin_config,
            initial_joint_states=initial_joint_states, server_name=server_name, ns=ns,
            verbose=verbose, output=output, launch_new_term=launch_new_term)
        return ros_port, mujoco_pid

    def _launch_roscore(self, port=None, set_new_master_vars=False):
        ros_port, _ = ros_common.launch_roscore(port=int(port) if port is not None else None, set_new_master_vars=set_new_master_vars)
        ros_common.change_ros_master(ros_port)
        return ros_port
