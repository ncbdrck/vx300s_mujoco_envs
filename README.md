# vx300s_mujoco_reach

A standalone test package: the ViperX-300 S **reach** task on the **MuJoCo backend**
(`mujoco_ros_pkgs`), mirroring the existing Gazebo VX300S reach. It exists to validate the
MultiROS MuJoCo backend end-to-end **without modifying `rl_environments`**.

It reuses, unchanged:
- the robot description `viperx300s_description` (`vx300s.urdf.xacro`, which already declares
  `EffortJointInterface` transmissions; the MuJoCo backend strips the gripper/finger transmissions
  automatically via `controlled_joints`, since they have no joints in the MJCF);
- the task parameters `rl_environments/config/vx300s_reach_task_config.yaml`;
- the training config `rl_training_validation/config/vx300s_reacher_td3.yaml`.

The arm is driven purely through the `ros_control` trajectory interface
(`/vx300s/arm_controller/command`); **MoveIt is not used** (end-effector poses come from forward
kinematics, joint state from `/vx300s/joint_states`).

## Layout
- `assets/vx300s_mjcf/` — the Trossen VX300S model from MuJoCo Menagerie, with its native MJCF
  actuators removed (joints are driven by `mujoco_ros_control`). `vx300s_scene.xml` is the scene
  the server loads (arm + ground plane).
- `config/vx300s_mujoco_control.yaml` — ros_control controllers + `mujoco_ros_control/pid_gains`.
- `config/vx300s_mujoco_plugins.yaml` — the `MujocoPlugins` block that loads the ros_control bridge.
- `config/initial_joint_states.yaml` — home configuration applied on load/reset.
- `launch/vx300s_mujoco_reach.launch` — brings up the URDF, the mujoco_ros server, and the controllers.
- `src/vx300s_mujoco_reach/` — the robot env and reach task env.

## Build
```bash
cd ~/rl_ws          # or your workspace root
catkin build vx300s_mujoco_reach
source devel/setup.bash
```
Prerequisite: the MuJoCo backend must be built (MuJoCo 3.3.5 + `mujoco_ros_pkgs` in the workspace;
`MUJOCO_DIR`/`LD_LIBRARY_PATH` exported). See the multiros installer's `-m` option.

## Run — one command (self-launch, default)

`gym.make` brings up roscore + the MuJoCo server + controllers itself, so a single command runs
everything (same one-call workflow as the `rl_environments` envs):
```bash
rosrun vx300s_mujoco_reach vx300s_mujoco_reach_test.py
```
Equivalently, in Python:
```python
import uniros as gym
env = gym.make("VX300SMujocoReacherSim-v0")   # launch_mujoco / new_roscore / load_robot default True
obs, info = env.reset(seed=0)
for _ in range(50):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        obs, info = env.reset()
env.close()
```
The training / validation scripts self-launch the same way; pass `--attach` to opt out.

## Run — attach mode (for debugging the bring-up)

Bring the simulation + controllers up with the launch file, then attach the RL env to it.

Terminal 1 — sim + controllers:
```bash
roslaunch vx300s_mujoco_reach vx300s_mujoco_reach.launch gui:=true
```
Check the stack is up:
```bash
rosservice list | grep /mujoco_server      # set_pause, reset, step, get_sim_info, ...
rostopic echo -n1 /vx300s/joint_states     # joint state is streaming
rostopic list | grep arm_controller        # /vx300s/arm_controller/command + /state
```
Terminal 2 — attach the env (set `ATTACH_MODE = True` in the test script, or `--attach` for
train/validate):
```python
import uniros as gym
env = gym.make("VX300SMujocoReacherSim-v0",
               launch_mujoco=False, new_roscore=False, load_robot=False)
```

Real-time (§7) vs paused MDP loop is selected with `realtime_mode=True/False` in either mode.
In the real-time loop the arm keeps moving continuously (physics never pauses) and observations
refresh between steps.

## Train
With the launch file running (attach mode), point the existing SB3 TD3 config at this env id:
```bash
rosrun rl_training_validation <train_script>.py \
    --env VX300SMujocoReacherSim-v0 \
    --config $(rospack find rl_training_validation)/config/vx300s_reacher_td3.yaml
```
(Use the train entry point / flags your `rl_training_validation` scripts expect; the env id is
`VX300SMujocoReacherSim-v0`.)

## Notes / known first-run checks
- `control_period` (0.001 s) must be >= the MJCF `<option timestep>`; the plugin warns otherwise.
- If the arm droops or oscillates under torque tracking, tune `mujoco_ros_control/pid_gains` in
  `config/vx300s_mujoco_control.yaml`.
- Standard reach only for now; the goal-conditioned (TD3+HER) variant is a follow-up.
