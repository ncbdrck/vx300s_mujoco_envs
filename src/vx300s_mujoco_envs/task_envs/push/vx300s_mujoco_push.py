#!/usr/bin/env python3

from typing import Any, Optional, Dict

import rospy
import rospkg
import numpy as np
from gymnasium import spaces
from urdf_parser_py.urdf import URDF
from gymnasium.envs.registration import register
import scipy.spatial

# Custom robot env
from vx300s_mujoco_envs.robot_envs import vx300s_mujoco_robot

# core modules of the framework
from multiros.utils import mujoco_core
from multiros.utils import mujoco_models
from multiros.utils import ros_common
from multiros.utils import ros_markers


register(
    id='VX300SMujocoPushSim-v0',
    entry_point='vx300s_mujoco_envs.task_envs.push.vx300s_mujoco_push:VX300SMujocoPushEnv',
    max_episode_steps=100,
)


class VX300SMujocoPushEnv(vx300s_mujoco_robot.VX300SMujocoRobotEnv):
    """
    Push task for the VX300S robot on the MuJoCo backend.

    The arm slides a cube resting on the table to a goal region on the same surface. Built on the
    same robot env and real-time loop as the reach task; the cube is a free-joint body in the MJCF
    scene that is repositioned each episode through the mujoco_ros set_body_state service (no
    per-episode spawn/delete). The goal is shown as an RViz marker.

    The task is done when the cube is within the success tolerance of the goal.

    Action Space - Continuous (6 arm-joint deltas).
    Observation  - Continuous (EE pos, cube pos, unit vector EE->cube, unit vector cube->goal,
                   distance cube->goal, joint values, previous action, joint velocities).

    Real-time vs normal MDP mode is selected by ``realtime_mode`` (see the reach task for details).
    """

    def __init__(self, launch_mujoco: bool = True, new_roscore: bool = True, roscore_port: str = None,
                 mujoco_paused: bool = False, mujoco_gui: bool = False, model_path: str = None,
                 model_pkg: str = "vx300s_mujoco_envs", model_name: str = "assets/vx300s_mjcf/vx300s_push_scene.xml",
                 server_name: str = "mujoco_server", seed: int = None, reward_type: str = "Dense",
                 delta_action: bool = True, delta_coeff: float = 0.05,
                 environment_loop_rate: float = 10, action_cycle_time: float = 0.100,
                 action_speed: float = 0.5, simple_dense_reward: bool = True,
                 log_internal_state: bool = False, debug: bool = False,
                 realtime_mode: bool = True, load_robot: bool = True,
                 sim_step_mode: int = 1, num_mujoco_steps: int = 1,
                 random_cube_spawn: bool = True, random_goal: bool = True,
                 cube_body_name: str = "cube"):

        # Real-time vs normal MDP step mode. Stored early so it's available when we decide
        # whether to register the rospy.Timer below.
        self.realtime_mode = realtime_mode
        self.random_cube_spawn = random_cube_spawn
        self.random_goal = random_goal
        self.cube_body_name = cube_body_name

        ros_port = None
        mujoco_pid = None

        # The mujoco_ros_control plugin config and the home pose. The launch file passes these to
        # the server; the self-launch path must pass them too, otherwise no controller_manager comes
        # up (and load_robot's controllers have nothing to attach to) and the home pose is wrong.
        _pkg_path = rospkg.RosPack().get_path("vx300s_mujoco_envs")
        plugin_config = _pkg_path + "/config/vx300s_mujoco_plugins.yaml"
        initial_joint_states = _pkg_path + "/config/initial_joint_states.yaml"

        # Launch the MuJoCo server (+ optional roscore)
        if launch_mujoco:
            ros_port, mujoco_pid = self._launch_mujoco(launch_roscore=new_roscore, port=roscore_port,
                                                       paused=mujoco_paused, headless=not mujoco_gui,
                                                       model_path=model_path, model_pkg=model_pkg,
                                                       model_name=model_name, server_name=server_name,
                                                       mujoco_plugin_config=plugin_config,
                                                       initial_joint_states=initial_joint_states)

        # Launch new roscore only
        elif new_roscore:
            ros_port = self._launch_roscore(port=roscore_port)

        # Attach to an already-running roscore
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

        # init the ros node
        if ros_port is not None:
            self.node_name = "VX300SMujocoPushEnvSim" + "_" + ros_port
        else:
            self.node_name = "VX300SMujocoPushEnvSim"

        rospy.init_node(self.node_name, anonymous=True)
        rospy.loginfo(f"Starting {self.node_name}")

        # In real-time mode the cached obs must be at least as fresh as one action cycle, so the
        # loop period must not exceed action_cycle_time. This constraint is irrelevant when not in
        # real-time mode (no background timer) or when action_cycle_time is 0 (deterministic-step
        # / fast mode, which advances the sim explicitly and never sleeps).
        if self.realtime_mode and action_cycle_time > 0.0 \
                and (1.0 / environment_loop_rate) > action_cycle_time:
            rospy.logerr("The environment loop rate is greater than the action cycle time. Exiting the program!")
            rospy.signal_shutdown("Exiting the program!")
            exit()

        self.log_internal_state = log_internal_state

        if reward_type.lower() == "sparse":
            self.reward_arc = "Sparse"
        elif reward_type.lower() == "dense":
            self.reward_arc = "Dense"
        else:
            rospy.logwarn(f"The given reward architecture '{reward_type}' not found. Defaulting to Dense!")
            self.reward_arc = "Dense"

        self.simple_dense_reward = simple_dense_reward

        self.delta_action = delta_action
        self.delta_coeff = delta_coeff
        self.action_cycle_time = action_cycle_time
        self.action_speed = action_speed
        self.debug = debug

        # load task parameters onto the parameter server
        ros_common.ros_load_yaml(pkg_name="vx300s_mujoco_envs", file_name="vx300s_push_task_config.yaml", ns="/")
        self._get_params()

        # Joint action space (6 arm joints)
        self.action_space = spaces.Box(low=np.array(self.min_joint_values), high=np.array(self.max_joint_values),
                                       dtype=np.float32)

        # ---- observation space pieces
        observations_high_ee_pos = np.array([self.position_ee_max["x"], self.position_ee_max["y"],
                                             self.position_ee_max["z"]])
        observations_low_ee_pos = np.array([self.position_ee_min["x"], self.position_ee_min["y"],
                                            self.position_ee_min["z"]])
        observations_high_cube_pos = np.array([self.position_cube_max["x"], self.position_cube_max["y"],
                                               self.position_cube_max["z"]])
        observations_low_cube_pos = np.array([self.position_cube_min["x"], self.position_cube_min["y"],
                                              self.position_cube_min["z"]])
        observations_high_vec_ee_cube = np.array([1.0, 1.0, 1.0])
        observations_low_vec_ee_cube = np.array([-1.0, -1.0, -1.0])
        observations_high_vec_cube_goal = np.array([1.0, 1.0, 1.0])
        observations_low_vec_cube_goal = np.array([-1.0, -1.0, -1.0])
        observations_high_dist = np.array([self.max_distance])
        observations_low_dist = np.array([0.0])
        observations_high_joint_values = self.max_joint_angles.copy()
        observations_low_joint_values = self.min_joint_angles.copy()
        observations_high_prev_action = self.max_joint_values.copy()
        observations_low_prev_action = self.min_joint_values.copy()
        observations_high_joint_vel = self.max_joint_vel.copy()
        observations_low_joint_vel = self.min_joint_vel.copy()

        high = np.concatenate(
            [observations_high_ee_pos, observations_high_cube_pos, observations_high_vec_ee_cube,
             observations_high_vec_cube_goal, observations_high_dist,
             observations_high_joint_values, observations_high_prev_action, observations_high_joint_vel, ])
        low = np.concatenate(
            [observations_low_ee_pos, observations_low_cube_pos, observations_low_vec_ee_cube,
             observations_low_vec_cube_goal, observations_low_dist,
             observations_low_joint_values, observations_low_prev_action, observations_low_joint_vel, ])

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # ---- goal sampling space (where the cube must end up)
        high_goal_pos_range = np.array([self.position_goal_max["x"], self.position_goal_max["y"],
                                        self.position_goal_max["z"]])
        low_goal_pos_range = np.array([self.position_goal_min["x"], self.position_goal_min["y"],
                                       self.position_goal_min["z"]])
        self.goal_space = spaces.Box(low=low_goal_pos_range, high=high_goal_pos_range, dtype=np.float32, seed=seed)

        # ---- cube spawn space (narrow reset region, distinct from the wide cube observation bounds)
        high_cube_range = np.array([self.cube_spawn_max["x"], self.cube_spawn_max["y"],
                                    self.cube_spawn_max["z"]])
        low_cube_range = np.array([self.cube_spawn_min["x"], self.cube_spawn_min["y"],
                                   self.cube_spawn_min["z"]])
        self.cube_space = spaces.Box(low=low_cube_range, high=high_cube_range, dtype=np.float32, seed=seed)

        # ---- workspace (for action validity)
        high_workspace_range = np.array([self.workspace_max["x"], self.workspace_max["y"], self.workspace_max["z"]])
        low_workspace_range = np.array([self.workspace_min["x"], self.workspace_min["y"], self.workspace_min["z"]])
        self.workspace_space = spaces.Box(low=low_workspace_range, high=high_workspace_range, dtype=np.float32)

        self.goal_marker = ros_markers.RosMarker(frame_id="world", ns="goal", marker_type=2, marker_topic="goal_pos",
                                                 lifetime=20.0)
        self.cube_marker = ros_markers.RosMarker(frame_id="world", ns="cube", marker_type=1, marker_topic="cube_pos",
                                                 lifetime=20.0)

        # VX300SMujocoRobotEnv maps real_time -> unpause_pause_physics; this single flag drives
        # whether the base env pauses physics around _set_action. load_robot=False attaches to a
        # robot/controller stack already brought up by the launch file.
        super().__init__(ros_port=ros_port, mujoco_pid=mujoco_pid, server_name=server_name, seed=seed,
                         real_time=self.realtime_mode, action_cycle_time=action_cycle_time,
                         load_robot=load_robot, sim_step_mode=sim_step_mode,
                         num_mujoco_steps=num_mujoco_steps)

        self.environment_loop_time = 1.0 / environment_loop_rate
        self.prev_action = None
        self.cube_pos = np.zeros(3, dtype=np.float32)

        if environment_loop_rate is not None:
            self.obs_r = None
            self.reward_r = None
            self.terminated_r = False
            self.truncated_r = False
            self.info_r = {}
            self.current_action = None
            self.init_done = False

            if self.debug:
                self.loop_counter = 0
                self.action_counter = 0

            # Real-time mode: drive the env loop with a rospy.Timer (paper section 7). Normal mode
            # reuses the same obs_r/reward_r cache but computes synchronously in _set_action,
            # so the timer is not registered.
            if self.realtime_mode:
                rospy.Timer(rospy.Duration(1.0 / environment_loop_rate), self.environment_loop)

        # dense-reward bookkeeping
        self.action_not_in_limits = False
        self.movement_result = False
        self.within_goal_space = False

        rospy.loginfo(f"Finished Init of {self.node_name}")

    # -------------------------------------------------------
    #   Env hooks

    def _set_init_params(self, options: Optional[Dict[str, Any]] = None):
        """
        Reset to the home pose, reposition the cube in its spawn region, and sample a new goal.
        """
        if self.log_internal_state:
            rospy.loginfo("Initialising the init params!")

        self.init_pos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # stop the env loop until reset completes
        self.init_done = False
        self.current_action = None

        # move the robot to the home pose
        self.movement_result = self.move_arm_joints(self.init_pos, time_from_start=self.action_speed)
        if not self.movement_result and self.log_internal_state:
            rospy.logwarn("Homing failed!")

        # reposition the cube in its spawn region (no spawn/delete; the body already exists)
        if self.random_cube_spawn:
            cube_init = self._sample_box(self.cube_space)
        else:
            cube_init = np.array([0.30, 0.0, self.cube_spawn_min["z"]], dtype=np.float32)
        mujoco_models.mujoco_set_body_state(body_name=self.cube_body_name,
                                            pos_x=float(cube_init[0]), pos_y=float(cube_init[1]),
                                            pos_z=float(cube_init[2]),
                                            set_pose=True, set_twist=True, reset_qpos=False,
                                            server_name=self.server_name)
        self.cube_pos = np.asarray(cube_init, dtype=np.float32)

        # sample a goal on the table for the cube to be pushed to
        if self.random_goal:
            self.push_goal = self._sample_box(self.goal_space)
        else:
            self.push_goal = np.array([0.50, 0.0, self.position_goal_min["z"]], dtype=np.float32)

        self.goal_marker.set_position(position=self.push_goal)
        self.goal_marker.publish()

        # initial EE pos + joint values (needed for delta actions)
        self.ee_pos = np.asarray(self.get_ee_pose(), dtype=np.float32)
        self.joint_values = self.get_joint_angles()

        self.action_not_in_limits = False
        self.within_goal_space = True
        self.prev_action = self.init_pos.copy()

        # init real-time cache
        self.obs_r = None
        self.reward_r = None
        self.terminated_r = False
        self.truncated_r = False
        self.info_r = {}

        if self.debug:
            self.loop_counter = 0
            self.action_counter = 0

        self.init_done = True

    def _set_action(self, action):
        """
        Real-time mode: stash the action; the rospy.Timer-driven environment_loop executes it.
        Normal MDP mode: execute synchronously and clear the cache so the post-action world is
        sampled fresh after the base env's action_cycle_time sleep.
        """
        self.prev_action = action.copy()
        self.current_action = action.copy()

        if self.debug:
            self.action_counter = 0

        if not self.realtime_mode:
            self.obs_r = None
            self.reward_r = None
            self.terminated_r = None
            self.info_r = {}
            self.execute_action(action)

    def _get_observation(self):
        obs = None
        if self.obs_r is not None:
            obs = self.obs_r.copy()
        if obs is None:
            obs = self.sample_observation()
        return obs.copy()

    def _get_reward(self, info: Optional[Dict[str, Any]] = None):
        reward = None
        if self.reward_r is not None:
            reward = self.reward_r
        if reward is None:
            reward = self.calculate_reward()
        return reward

    def _compute_terminated(self, info: Optional[Dict[str, Any]] = None):
        terminated = self.terminated_r
        self.info = self.info_r

        if "is_success" not in self.info:
            self.info["is_success"] = bool(terminated)

        if terminated is None:
            terminated = self.check_if_done()
            self.info["is_success"] = bool(terminated)

        return terminated

    def _compute_truncated(self, info: Optional[Dict[str, Any]] = None):
        return self.truncated_r

    # -------------------------------------------------------
    #   Real-time loop + action execution

    def environment_loop(self, event):
        """
        Real-time RL loop (paper section 7). Periodically refreshes obs/reward/done and re-applies
        the latest action (action repeats) so the agent never waits on sensor/actuator processing.
        """
        if self.init_done:
            if rospy.is_shutdown():
                return
            jv = getattr(self, "joint_values", None)
            if jv is None or len(jv) < 6:
                return

            if self.debug:
                self.loop_counter += 1

            self.info_r = {}
            self.obs_r = self.sample_observation()
            self.reward_r = self.calculate_reward()
            self.terminated_r = self.check_if_done()

            if self.current_action is not None:
                self.execute_action(self.current_action)
                if self.debug:
                    self.action_counter += 1

    def execute_action(self, action):
        """
        Apply a joint-delta action: add the (scaled) action to the current joints, clip to
        limits, run the workspace + per-link safety checks, then publish a joint trajectory.
        """
        # delta action on top of current joints
        if self.delta_action:
            self.joint_values = self.get_joint_angles()
            if self.joint_values is None or len(self.joint_values) < len(self.min_joint_values):
                if self.log_internal_state:
                    rospy.logwarn("Joint action rejected: current joint vector is stale or empty.")
                self.movement_result = False
                self.within_goal_space = False
                return
            action = np.asarray(self.joint_values) + (np.asarray(action) * self.delta_coeff)

        min_joint_values = np.array(self.min_joint_values)
        max_joint_values = np.array(self.max_joint_values)
        self.action_not_in_limits = np.any(action <= (min_joint_values + 0.0001)) or np.any(
            action >= (max_joint_values - 0.0001))

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

    def get_cube_pose(self):
        """
        Read the cube position from the MuJoCo server. Falls back to the last known position if
        the service call fails.
        """
        _, pose, _, success = mujoco_models.mujoco_get_body_state(body_name=self.cube_body_name,
                                                                  server_name=self.server_name)
        if success:
            return np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float32)
        return self.cube_pos

    def sample_observation(self):
        """
        Build the observation vector: EE pos, cube pos, unit vector EE->cube, unit vector
        cube->goal, distance cube->goal, joint positions, previous action, joint velocities.
        """
        ee = self.fk_pykdl(self.get_joint_angles())
        if ee is None:
            ee = self.ee_pos
        self.ee_pos = np.asarray(ee, dtype=np.float32)

        self.cube_pos = self.get_cube_pose()
        self.cube_marker.set_position(position=self.cube_pos)
        self.cube_marker.publish()

        vec_ee_cube = self._safe_unit_vector(self.cube_pos - self.ee_pos)
        vec_cube_goal = self._safe_unit_vector(self.push_goal - self.cube_pos)
        euclidean_distance_cube_goal = scipy.spatial.distance.euclidean(self.cube_pos, self.push_goal)

        self.joint_values = list(self.joint_pos_all)

        if self.prev_action is None:
            prev_action = self.get_joint_angles()
        else:
            prev_action = self.prev_action.copy()

        obs = np.concatenate((self.ee_pos, self.cube_pos, vec_ee_cube, vec_cube_goal,
                              euclidean_distance_cube_goal, self.joint_pos_all, prev_action,
                              self.current_joint_velocities), axis=None, dtype=np.float32)

        return obs.copy()

    def calculate_reward(self):
        """
        Sparse: +1 when the cube is at the goal, -1 otherwise.
        Dense (simple): negative (distance EE->cube + distance cube->goal), so the agent is
        rewarded for both approaching the cube and pushing it toward the goal.
        """
        reward = 0.0
        achieved_goal = self.cube_pos
        desired_goal = self.push_goal

        if self.reward_arc == "Sparse":
            reward = -1.0
            self.goal_marker.set_color(r=1.0, g=0.0)
            self.goal_marker.set_duration(duration=5)
            if self.check_if_reach_done(achieved_goal, desired_goal):
                reward = 1.0
                self.goal_marker.set_color(r=0.0, g=1.0)
            self.goal_marker.publish()
        else:
            if self.simple_dense_reward:
                dist_ee_cube = scipy.spatial.distance.euclidean(self.ee_pos, self.cube_pos)
                dist_cube_goal = scipy.spatial.distance.euclidean(self.cube_pos, desired_goal)
                reward += - (dist_ee_cube + dist_cube_goal)
            else:
                done = self.check_if_reach_done(achieved_goal, desired_goal)
                if done:
                    reward += self.reached_goal_reward
                    self.goal_marker.set_color(r=0.0, g=1.0)
                    self.goal_marker.set_duration(duration=30)
                else:
                    self.goal_marker.set_color(r=1.0, g=0.0)
                    self.goal_marker.set_duration(duration=5)
                    dist_ee_cube = scipy.spatial.distance.euclidean(self.ee_pos, self.cube_pos)
                    dist_cube_goal = scipy.spatial.distance.euclidean(self.cube_pos, desired_goal)
                    reward += - self.mult_dist_reward * (dist_ee_cube + dist_cube_goal)
                    reward += self.step_reward
                self.goal_marker.publish()
                reward += self.action_not_in_limits * self.joint_limits_reward
                reward += (not self.within_goal_space) * self.not_within_goal_space_reward
                if not self.movement_result:
                    reward += self.none_exe_reward

        return reward

    def check_if_done(self):
        """
        Done when the cube is within the success tolerance of the goal.
        """
        done_push = self.check_if_reach_done(self.cube_pos, self.push_goal)
        if done_push:
            done = True
            self.current_action = None
            self.init_done = False
            self.info_r['is_success'] = True
        else:
            done = False
            self.info_r['is_success'] = False
        return done

    def check_if_reach_done(self, achieved_goal, desired_goal):
        # Push success is measured in the table plane (x, y); the cube z stays on the surface.
        distance = scipy.spatial.distance.euclidean(np.asarray(achieved_goal)[:2], np.asarray(desired_goal)[:2])
        return distance <= self.reach_tolerance

    def check_action_within_workspace(self, action):
        """
        True if the FK of the action lands inside the workspace box.
        """
        ee_pos = self.fk_pykdl(action=action)
        if ee_pos is not None:
            return bool(self.workspace_space.contains(ee_pos))
        return False

    # Controlled arm joints, in mujoco_ros_control registration / joint_states order.
    ARM_JOINT_NAMES = ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")

    def _arm_joint_limits_from_urdf(self):
        """
        Read (lower, upper, velocity) limits for the 6 controlled arm joints from the manufacturer
        description, in arm-joint (transmission / joint_states) order.

        The URDF is processed directly from the package (``param_name=None`` -> returned as a string,
        not set on the param server) rather than read from the ``robot_description`` parameter,
        because ``_get_params`` runs during construction before ``super().__init__()`` has loaded
        the description in the self-launch path, so the param may not exist yet. Joint limits are
        identical with or without transmission stripping, so the raw description is fine here.

        Returns:
            (lower, upper, vel): three lists of 6 floats each.
        """
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
        """
        Load task parameters from the parameter server (set by vx300s_push_task_config.yaml).
        """
        self.min_joint_values = rospy.get_param('/vx300s/min_joint_pos')
        self.max_joint_values = rospy.get_param('/vx300s/max_joint_pos')

        self.position_ee_max = rospy.get_param('/vx300s/position_ee_max')
        self.position_ee_min = rospy.get_param('/vx300s/position_ee_min')
        self.max_distance = rospy.get_param('/vx300s/max_distance')

        # Joint-position and joint-velocity observation bounds for the 6 controlled arm joints,
        # in the order mujoco_ros_control registers them from the URDF transmissions (which is the
        # order /vx300s/joint_states publishes).
        lower, upper, vel = self._arm_joint_limits_from_urdf()
        self.min_joint_angles = lower
        self.max_joint_angles = upper
        self.min_joint_vel = [-v for v in vel]
        self.max_joint_vel = list(vel)

        # Cube observation bounds (wide) and the cube spawn region (narrow) are separate: a pushed
        # cube travels into the goal region, so the observation box must be wider than the spawn box.
        self.position_cube_max = rospy.get_param('/vx300s/position_cube_max')
        self.position_cube_min = rospy.get_param('/vx300s/position_cube_min')
        self.cube_spawn_max = rospy.get_param('/vx300s/cube_spawn_max')
        self.cube_spawn_min = rospy.get_param('/vx300s/cube_spawn_min')

        self.position_goal_max = rospy.get_param('/vx300s/position_goal_max')
        self.position_goal_min = rospy.get_param('/vx300s/position_goal_min')

        self.reach_tolerance = rospy.get_param('/vx300s/reach_tolerance')

        self.step_reward = rospy.get_param('/vx300s/step_reward')
        self.mult_dist_reward = rospy.get_param('/vx300s/multiplier_dist_reward')
        self.reached_goal_reward = rospy.get_param('/vx300s/reached_goal_reward')
        self.joint_limits_reward = rospy.get_param('/vx300s/joint_limits_reward')
        self.none_exe_reward = rospy.get_param('/vx300s/none_exe_reward')
        self.not_within_goal_space_reward = rospy.get_param('/vx300s/not_within_goal_space_reward')

        self.workspace_max = rospy.get_param('/vx300s/workspace_max')
        self.workspace_min = rospy.get_param('/vx300s/workspace_min')

    # -------------------------------------------------------
    #   Launch helpers

    def _launch_mujoco(self, launch_roscore=True, port=None, paused=False, use_sim_time=True,
                       model_path=None, model_pkg=None, model_name=None, headless=True,
                       no_render=False, realtime=None, mujoco_plugin_config=None,
                       initial_joint_states=None, server_name="mujoco_server", ns="",
                       verbose=False, output='screen', launch_new_term=True):
        """
        Launch a MuJoCo server for the push scene.
        """
        ros_port, mujoco_pid = mujoco_core.launch_mujoco(
            launch_roscore=launch_roscore, port=port, paused=paused, use_sim_time=use_sim_time,
            model_path=model_path, model_pkg=model_pkg, model_name=model_name, headless=headless,
            no_render=no_render, realtime=realtime, mujoco_plugin_config=mujoco_plugin_config,
            initial_joint_states=initial_joint_states, server_name=server_name, ns=ns,
            verbose=verbose, output=output, launch_new_term=launch_new_term)
        return ros_port, mujoco_pid

    def _launch_roscore(self, port=None, set_new_master_vars=False):
        ros_port, _ = ros_common.launch_roscore(port=int(port), set_new_master_vars=set_new_master_vars)
        ros_common.change_ros_master(ros_port)
        return ros_port
