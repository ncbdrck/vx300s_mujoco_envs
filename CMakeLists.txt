cmake_minimum_required(VERSION 3.0.2)
project(vx300s_mujoco_envs)

find_package(catkin REQUIRED COMPONENTS
  rospy
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS rospy
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

# Make the test / train / validate scripts runnable with rosrun.
catkin_install_python(PROGRAMS
  scripts/vx300s_mujoco_reach_test.py
  scripts/vx300s_mujoco_reach_train.py
  scripts/vx300s_mujoco_reach_validate.py
  scripts/vx300s_mujoco_push_test.py
  scripts/vx300s_mujoco_push_train.py
  scripts/vx300s_mujoco_push_validate.py
  scripts/vx300s_mujoco_push_poke.py
  scripts/vx300s_mujoco_pnp_test.py
  scripts/vx300s_mujoco_pnp_train.py
  scripts/vx300s_mujoco_pnp_validate.py
  scripts/vx300s_mujoco_pnp_poke.py
  scripts/vx300s_mujoco_reach_goal_test.py
  scripts/vx300s_mujoco_reach_goal_train.py
  scripts/vx300s_mujoco_reach_goal_validate.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install Python-side resources.
install(DIRECTORY launch config assets
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
