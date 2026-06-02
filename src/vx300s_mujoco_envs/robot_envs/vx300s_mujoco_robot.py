#!/usr/bin/env python3

import numpy as np

import rospy
import rostopic
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from gymnasium.envs.registration import register

from multiros.envs import MujocoBaseEnv
from multiros.utils import mujoco_core
from multiros.utils import ros_common
from multiros.utils import ros_kinematics

from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_kinematics import KDLKinematics
from tf.transformations import euler_from_quaternion


register(
    id='VX300SMujocoRobotEnv-v0',
    entry_point='vx300s_mujoco_envs.robot_envs.vx300s_mujoco_robot:VX300SMujocoRobotEnv',
    max_episode_steps=1000,
)


class VX300SMujocoRobotEnv(MujocoBaseEnv.MujocoBaseEnv):
    """
    Superclass for all VX300S MuJoCo Robot environments.

    Mirrors the Gazebo VX300S robot env, but drives the arm purely through the ros_control
    trajectory interface (no MoveIt): the joint state is read from /vx300s/joint_states and
    commands are published to /vx300s/arm_controller/command. End-effector poses and the
    per-link safety check use forward kinematics from the robot description.

    Sensor Topic List:
        /vx300s/joint_states : JointState of the robot joints.

    Actuators Topic List:
        /vx300s/arm_controller/command     : arm joint trajectory commands.
        /vx300s/gripper_controller/command : gripper joint trajectory commands.
    """

    # Default (arm-only) controller + joint sets, used by reach and push. Tasks that also drive the
    # gripper (e.g. pick-and-place) pass the gripper-enabled variants below.
    ARM_CONTROLLERS = ["joint_state_controller", "arm_controller"]
    ARM_JOINTS = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
    ARM_GRIPPER_CONTROLLERS = ["joint_state_controller", "arm_controller", "gripper_controller"]
    ARM_GRIPPER_JOINTS = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle",
                          "wrist_rotate", "left_finger", "right_finger"]

    def __init__(self, ros_port: str = None, mujoco_pid=None, server_name: str = "mujoco_server",
                 seed: int = None, real_time: bool = False, action_cycle_time: float = 0.0,
                 load_robot: bool = True, sim_step_mode: int = 1, num_mujoco_steps: int = 1,
                 controllers_file: str = "vx300s_mujoco_control.yaml",
                 controllers_list: list = None, controlled_joints: list = None,
                 readiness_timeout: float = 30.0):

        rospy.loginfo("Start Init VX300SMujocoRobotEnv")

        # How long the per-topic readiness checks wait before raising
        # (covers slower hardware / GUI startup). Set higher with the
        # constructor kwarg if your launch takes longer.
        self.readiness_timeout = float(readiness_timeout)

        if ros_port is not None:
            ros_common.change_ros_master(ros_port=ros_port)

        self.real_time = real_time
        self.server_name = server_name

        # In real-time mode the simulation runs free (no pause/unpause around the action);
        # otherwise the base env pauses physics around _set_action.
        if self.real_time:
            unpause_pause_physics = False
        else:
            unpause_pause_physics = True

        if not self.real_time:
            mujoco_core.unpause_mujoco(server_name=self.server_name)

        # Robot bring-up: load the URDF to the parameter server, start the robot_state_publisher
        # and spawn the controllers. The robot geometry itself comes from the MJCF scene the
        # mujoco_ros server loaded at launch. Set load_robot=False to attach to a stack that a
        # launch file already brought up (the recommended first-run path).
        spawn_robot = load_robot
        # The manufacturer xacro is reused unmodified. It declares transmissions for the gripper
        # and fingers, which have no joints in the MJCF; the backend strips those automatically
        # because we pass controlled_joints below (mujoco_ros_control would otherwise abort).
        urdf_pkg_name = "viperx300s_description"
        urdf_file_name = "vx300s.urdf.xacro"
        urdf_folder = "/urdf"
        urdf_xacro_args = None

        namespace = "/vx300s"

        robot_state_publisher_max_freq = None
        new_robot_state_term = False

        # Controllers + driven joints are chosen by the task env. Reach/push leave these None and
        # get the arm-only defaults (no gripper controller spawned); pick-and-place passes the
        # gripper-enabled sets so left_finger/right_finger are driven too. The backend keeps only
        # the controlled joints' URDF transmissions and strips the rest (mujoco_ros_control would
        # otherwise abort on a transmission whose joint is absent from the MJCF).
        controller_package_name = "vx300s_mujoco_envs"
        if controllers_list is None:
            controllers_list = self.ARM_CONTROLLERS
        if controlled_joints is None:
            controlled_joints = self.ARM_JOINTS
        # Remember which controllers this task spawned so the readiness check can wait on the right
        # ones (e.g. the gripper controller for pick-and-place, absent for reach/push).
        self.controllers_list = controllers_list

        reset_controllers = False

        # sim_step_mode 1 = unpause / act / pause (default, wall-clock paced). Mode 2 uses the
        # deterministic /<server>/step action (sim must be paused) and advances num_mujoco_steps
        # physics ticks per env step with no wall-clock sleep -> runs as fast as the CPU allows.
        # Both are passed in by the task env so an env author picks the regime without editing here.

        mujoco_max_update_rate = None
        mujoco_timestep = None

        kill_rosmaster = True
        kill_mujoco = True
        clean_logs = False

        super().__init__(
            spawn_robot=spawn_robot, urdf_pkg_name=urdf_pkg_name, urdf_file_name=urdf_file_name,
            urdf_folder=urdf_folder, urdf_xacro_args=urdf_xacro_args, namespace=namespace,
            robot_state_publisher_max_freq=robot_state_publisher_max_freq, new_robot_state_term=new_robot_state_term,
            controllers_file=controllers_file, controllers_list=controllers_list,
            reset_controllers=reset_controllers, sim_step_mode=sim_step_mode,
            num_mujoco_steps=num_mujoco_steps, mujoco_max_update_rate=mujoco_max_update_rate,
            mujoco_timestep=mujoco_timestep, kill_rosmaster=kill_rosmaster, kill_mujoco=kill_mujoco,
            clean_logs=clean_logs, ros_port=ros_port, mujoco_pid=mujoco_pid, server_name=server_name, seed=seed,
            unpause_pause_physics=unpause_pause_physics, action_cycle_time=action_cycle_time,
            controller_package_name=controller_package_name, controlled_joints=controlled_joints)

        # ---------- joint state
        if namespace is not None and namespace != '/':
            self.joint_state_topic = namespace + "/joint_states"
        else:
            self.joint_state_topic = "/joint_states"

        self.joint_state_sub = rospy.Subscriber(self.joint_state_topic, JointState, self.joint_state_callback)
        self.joint_state = JointState()
        self.joint_pos_all = []
        self.current_joint_velocities = []
        self.current_joint_efforts = []

        self._check_connection_and_readiness()

        self.arm_joint_names = ["waist",
                                "shoulder",
                                "elbow",
                                "forearm_roll",
                                "wrist_angle",
                                "wrist_rotate"]

        self.gripper_joint_names = ["left_finger",
                                    "right_finger"]

        # ---------- low-level trajectory controllers
        self.arm_controller_pub = rospy.Publisher('/vx300s/arm_controller/command',
                                                  JointTrajectory,
                                                  queue_size=10)

        self.gripper_controller_pub = rospy.Publisher('/vx300s/gripper_controller/command',
                                                      JointTrajectory,
                                                      queue_size=10)

        # ---------- kinematics (forward kinematics for EE pose + safety; no MoveIt)
        self.ee_link = "vx300s/ee_gripper_link"
        self.ref_frame = "vx300s/base_link"

        self.pykdl_robot = URDF.from_parameter_server(key='vx300s/robot_description')
        self.kdl_kin = KDLKinematics(urdf=self.pykdl_robot, base_link=self.ref_frame, end_link=self.ee_link)

        self.ros_kin = ros_kinematics.Kinematics_pyrobot(robot_description_parm="vx300s/robot_description",
                                                         base_link=self.ref_frame,
                                                         end_link=self.ee_link)

        # Per-link FK chains for the table-collision safety check.
        self._safety_kin = {}
        for _link in self.SAFETY_CHECK_LINKS:
            try:
                _kin = KDLKinematics(urdf=self.pykdl_robot,
                                     base_link=self.ref_frame,
                                     end_link=_link)
                self._safety_kin[_link] = (_kin, int(_kin.num_joints))
            except Exception as _e:
                rospy.logwarn(f"[SAFETY] kinematics setup failed for {_link}: {_e}")

        if not self.real_time:
            mujoco_core.pause_mujoco(server_name=self.server_name)
        else:
            mujoco_core.unpause_mujoco(server_name=self.server_name)

        rospy.loginfo("End Init VX300SMujocoRobotEnv")

    # ---------------------------------------------------
    #   Custom methods for the Custom Robot Environment

    def fk_pykdl(self, action):
        """
        Calculate the forward kinematics (end-effector position) for a joint configuration.

        Args:
            action: joint positions of the robot arm (in radians)

        Returns:
            ee_position: end-effector position as a numpy array, or None if the input is empty.
        """
        if action is None or len(action) == 0:
            return None

        pose = self.kdl_kin.forward(action)
        ee_position = np.array([pose[0, 3], pose[1, 3], pose[2, 3]], dtype=np.float32)
        return ee_position

    def calculate_fk(self, joint_positions, euler=True):
        """
        Calculate the forward kinematics of the robot arm using the ros_kinematics package.

        Returns:
            done, ee_position, ee_orientation
        """
        done, ee_position, ee_ori = self.ros_kin.calculate_fk(joint_positions, des_frame=self.ee_link, euler=euler)
        return done, ee_position, ee_ori

    def calculate_ik(self, target_pos, ee_ori=np.array([0.0, 0.0, 0.0, 1.0])):
        """
        Calculate the inverse kinematics of the robot arm using the ros_kinematics package.

        Returns:
            done, joint_positions
        """
        target_pose = np.concatenate((target_pos, ee_ori))
        current_joints = self.get_joint_angles()
        done, joint_positions = self.ros_kin.calculate_ik(target_pose=target_pose, tolerance=[1e-3] * 6,
                                                          init_joint_positions=current_joints)
        return done, joint_positions

    # Arm links whose world z must stay above the table for the action to be safe. Order
    # matches the URDF chain shoulder -> ee_gripper; downstream gripper links are covered
    # implicitly by checking gripper_link / ee_gripper_link.
    SAFETY_CHECK_LINKS = (
        "vx300s/shoulder_link",
        "vx300s/upper_arm_link",
        "vx300s/upper_forearm_link",
        "vx300s/lower_forearm_link",
        "vx300s/wrist_link",
        "vx300s/gripper_link",
        "vx300s/ee_gripper_link",
    )

    def _check_action_links_safe(self, joint_targets, current_joints=None):
        """
        Predict each arm link's world z under ``joint_targets`` and reject the action if any
        link would dip below ``table_z + safety_z_margin``. Also caps |target - current| per
        joint at ``max_joint_delta``.

        Returns:
            (safe, reason): safe is True if every link stays above the floor and no joint
            exceeds the per-step delta cap; reason names the first failure otherwise.
        """
        table_z = float(rospy.get_param("/vx300s/table_z", -0.005))
        margin = float(rospy.get_param("/vx300s/safety_z_margin", 0.015))
        max_delta = float(rospy.get_param("/vx300s/max_joint_delta", 0.5))
        floor = table_z + margin

        q = np.asarray(joint_targets, dtype=np.float64)

        if current_joints is not None:
            cur = np.asarray(current_joints, dtype=np.float64)
            if cur.shape == q.shape:
                deltas = np.abs(q - cur)
                if np.any(deltas > max_delta):
                    idx = int(np.argmax(deltas))
                    return False, f"joint[{idx}] delta {deltas[idx]:.3f} > {max_delta}"

        for link, (kin, n) in self._safety_kin.items():
            try:
                pose = kin.forward(q[:n])
            except Exception as e:
                return False, f"FK failed for {link}: {e}"
            z = float(pose[2, 3])
            if z < floor:
                return False, f"{link} predicted z={z:.3f} < floor={floor:.3f}"

        return True, None

    def joint_state_callback(self, joint_state):
        """
        Store the latest joint state of the robot.
        """
        if joint_state is not None:
            self.joint_state = joint_state
            self.joint_state_names = list(joint_state.name)
            self.joint_pos_all = list(joint_state.position)
            self.current_joint_velocities = list(joint_state.velocity)
            self.current_joint_efforts = list(joint_state.effort)

    def _wait_for_joint_convergence(self, joint_names, target_positions,
                                    tolerance: float = 0.01,
                                    timeout: float = 2.0,
                                    poll: float = 0.02) -> bool:
        """
        Block until every joint in ``joint_names`` is within ``tolerance`` of the matching entry
        in ``target_positions``, or until ``timeout`` seconds. Returns True on convergence, False
        on timeout — so the caller can detect a stalled controller and not silently treat
        publish-and-return as "command applied". A default tolerance of 0.01 (rad for arm joints,
        m for prismatic fingers) is loose enough not to trip on the trajectory controller's last
        few percent settling but tight enough to confirm the gross motion happened.
        """
        deadline = rospy.get_time() + float(timeout)
        targets = list(target_positions)
        while rospy.get_time() < deadline and not rospy.is_shutdown():
            if self.joint_state_names and self.joint_pos_all is not None:
                name_to_pos = dict(zip(self.joint_state_names, self.joint_pos_all))
                try:
                    errs = [abs(name_to_pos[n] - t) for n, t in zip(joint_names, targets)]
                except KeyError:
                    rospy.sleep(poll)
                    continue
                if errs and max(errs) <= tolerance:
                    return True
            rospy.sleep(poll)
        return False

    def move_arm_joints(self, q_positions: np.ndarray, time_from_start: float = 0.5,
                        await_convergence: bool = False,
                        convergence_tolerance: float = 0.01) -> bool:
        """
        Command the arm joints with a single-point joint trajectory.

        With ``await_convergence=True`` the call blocks until the arm joints reach
        ``q_positions`` within ``convergence_tolerance`` rad, or returns False after roughly
        twice ``time_from_start``. Default is False (fire-and-forget) so the per-tick action
        path is unaffected; reset paths pass True so the env doesn't sample obs from a
        mid-motion arm.
        """
        trajectory = JointTrajectory()
        trajectory.joint_names = self.arm_joint_names
        trajectory.points.append(JointTrajectoryPoint())
        trajectory.points[0].positions = q_positions
        trajectory.points[0].velocities = [0.0] * len(self.arm_joint_names)
        trajectory.points[0].accelerations = [0.0] * len(self.arm_joint_names)
        trajectory.points[0].time_from_start = rospy.Duration(time_from_start)

        self.arm_controller_pub.publish(trajectory)
        if not await_convergence:
            return True
        return self._wait_for_joint_convergence(
            self.arm_joint_names, q_positions,
            tolerance=convergence_tolerance,
            timeout=max(2.0 * time_from_start, 1.0),
        )

    def move_gripper_joints(self, q_positions: np.ndarray, time_from_start: float = 0.5,
                            await_convergence: bool = False,
                            convergence_tolerance: float = 0.005) -> bool:
        """
        Command the gripper joints with a single-point joint trajectory.

        With ``await_convergence=True`` the call blocks until the gripper joints reach
        ``q_positions`` within ``convergence_tolerance`` m, or returns False after roughly
        twice ``time_from_start``. Default is False; reset and explicit open/close paths
        pass True.
        """
        trajectory = JointTrajectory()
        trajectory.joint_names = self.gripper_joint_names
        trajectory.points.append(JointTrajectoryPoint())
        trajectory.points[0].positions = q_positions
        trajectory.points[0].velocities = [0.0] * len(self.gripper_joint_names)
        trajectory.points[0].accelerations = [0.0] * len(self.gripper_joint_names)
        trajectory.points[0].time_from_start = rospy.Duration(time_from_start)

        self.gripper_controller_pub.publish(trajectory)
        if not await_convergence:
            return True
        return self._wait_for_joint_convergence(
            self.gripper_joint_names, q_positions,
            tolerance=convergence_tolerance,
            timeout=max(2.0 * time_from_start, 1.0),
        )

    def get_ee_pose(self):
        """
        Return the end-effector position as a numpy array, computed from the current joint state.
        """
        ee = self.fk_pykdl(self.joint_pos_all[:len(self.arm_joint_names)] if self.joint_pos_all else None)
        if ee is None:
            return np.zeros(3, dtype=np.float32)
        return ee

    def get_ee_rpy(self):
        """
        Return the end-effector orientation (roll, pitch, yaw) computed from the current joints.
        """
        joints = self.get_joint_angles()
        done, _, ee_ori = self.calculate_fk(joints, euler=False)
        if not done or ee_ori is None:
            return np.zeros(3, dtype=np.float32)
        return np.array(euler_from_quaternion(ee_ori), dtype=np.float32)

    def get_joint_angles(self):
        """
        Return the current arm joint angles (6 elements) from the latest joint state.
        """
        if not self.joint_pos_all:
            return []
        return list(self.joint_pos_all[:len(self.arm_joint_names)])

    # ---------------------------------------------------
    #   Readiness

    def _wait_for_topic(self, topic, timeout=30.0, poll=0.2):
        """
        Block until ``topic`` is announced on the master, or raise after
        ``timeout`` seconds with an actionable error. Polling-based wrapper
        around ``rostopic.get_topic_type(..., blocking=False)`` so a broken
        launch never hangs indefinitely.
        """
        deadline = rospy.get_time() + timeout
        while rospy.get_time() < deadline and not rospy.is_shutdown():
            topic_type, _, _ = rostopic.get_topic_type(topic, blocking=False)
            if topic_type:
                rospy.logdebug(f"{topic} is up: {topic_type}")
                return True
            rospy.sleep(poll)
        raise RuntimeError(
            f"Readiness check timed out after {timeout:.0f}s waiting for {topic}. "
            "Confirm the MuJoCo server + controllers came up "
            "(check the launch output)."
        )

    def _check_joint_states_ready(self):
        self._wait_for_topic(self.joint_state_topic, timeout=self.readiness_timeout)
        return True

    def _check_ros_controllers_ready(self):
        # Wait on the state topic of every trajectory controller this task actually spawned. Reach
        # and push spawn only arm_controller; pick-and-place also spawns gripper_controller, which
        # is task-critical, so it must be verified too. (joint_state_controller has no /state topic;
        # its readiness is covered by _check_joint_states_ready via /joint_states.)
        for controller in getattr(self, "controllers_list", ["arm_controller"]):
            if controller == "joint_state_controller":
                continue
            self._wait_for_topic(f"/vx300s/{controller}/state",
                                 timeout=self.readiness_timeout)
        return True

    def _check_connection_and_readiness(self):
        """
        Check that the joint-state stream and the controllers are up before the task starts.
        """
        self._check_joint_states_ready()
        self._check_ros_controllers_ready()
        rospy.loginfo("All systems are ready!")
        return True
