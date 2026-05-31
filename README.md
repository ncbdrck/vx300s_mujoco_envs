# vx300s_mujoco_envs

> ⚠️ **Under development.** Part of the experimental MuJoCo backend for MultiROS/UniROS
> (`multiros` branch `feature/mujoco-backend`). APIs and structure may change; not yet merged
> into the stable release.

ViperX-300 S MuJoCo environments (`mujoco_ros_pkgs` backend), used to validate the MultiROS
MuJoCo backend end-to-end. The package mirrors the existing Gazebo VX300S envs but builds each
env entirely from the `multiros` MuJoCo tooling (`mujoco_core` / `mujoco_models` / `mujoco_physics`
/ `MujocoBaseEnv`), so creating a MuJoCo env follows the same workflow as a Gazebo one.

Currently implemented: **reach**. Planned: **pick-and-place**, **push** (added under
`task_envs/` as they are validated).

The arm is driven purely through the `ros_control` trajectory interface
(`/vx300s/arm_controller/command`); **MoveIt is not used** — end-effector poses come from forward
kinematics, joint state from `/vx300s/joint_states`.

It reuses the robot description `viperx300s_description` (`vx300s.urdf.xacro`); the MuJoCo backend
strips the gripper/finger transmissions automatically (via `controlled_joints`) since they have no
joints in the MJCF. Task and training configs are self-contained in this package's `config/`.

## Layout
- `assets/vx300s_mjcf/` — the Trossen VX300S model from MuJoCo Menagerie, with its native MJCF
  actuators removed (joints are driven by `mujoco_ros_control`). `vx300s_scene.xml` is the scene
  the server loads (arm + ground plane).
- `config/vx300s_mujoco_control.yaml` — ros_control controllers (effort JointTrajectoryController).
- `config/vx300s_mujoco_plugins.yaml` — the `MujocoPlugins` block that loads the ros_control bridge.
- `config/initial_joint_states.yaml` — home configuration applied on load/reset.
- `config/vx300s_reach_task_config.yaml` — task params (reward, workspace, goal sampling).
- `config/vx300s_reacher_sac.yaml`, `config/vx300s_reacher_td3.yaml` — SB3 training hyper-parameters.
- `launch/vx300s_mujoco_reach.launch` — brings up the URDF, the mujoco_ros server, and controllers.
- `src/vx300s_mujoco_envs/robot_envs/` — the shared VX300S robot env.
- `src/vx300s_mujoco_envs/task_envs/reach/` — the reach task env (env id `VX300SMujocoReacherSim-v0`).

## Build
```bash
cd ~/rl_ws          # or your workspace root
catkin_make         # this workspace uses catkin_make
source devel/setup.bash
```
Prerequisite: the MuJoCo backend must be built (MuJoCo 3.3.5 + `mujoco_ros_pkgs` in the workspace;
`MUJOCO_DIR`/`LD_LIBRARY_PATH` exported). See the multiros installer's `-m` option.

## Run — one command (self-launch, default)

`gym.make` brings up roscore + the MuJoCo server + controllers itself, so a single command runs
everything (same one-call workflow as the Gazebo `rl_environments` envs):
```bash
rosrun vx300s_mujoco_envs vx300s_mujoco_reach_test.py
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
roslaunch vx300s_mujoco_envs vx300s_mujoco_reach.launch gui:=true
```
Check the stack is up:
```bash
rosservice list | grep /mujoco_server      # set_pause, reset, step, get_sim_info, ...
rostopic echo -n1 /vx300s/joint_states     # joint state is streaming
rostopic list | grep arm_controller        # /vx300s/arm_controller/command + /state
```
Terminal 2 — attach the env (set `ATTACH_MODE = True` in the test script, or `--attach` for
train/validate).

## Train / validate

```bash
# Train (SAC by default; self-launches the sim). --no-realtime for the paused MDP loop,
# --fast for the deterministic fast-step regime, --steps N to override the config step count.
rosrun vx300s_mujoco_envs vx300s_mujoco_reach_train.py

# Validate a saved model:
rosrun vx300s_mujoco_envs vx300s_mujoco_reach_validate.py --episodes 20
```

Training regimes (set on the train script):
- **real-time** (default) — UniROS paper §7 loop; physics never pauses (sim→real fidelity).
- **`--no-realtime`** — paused MDP loop; every sample is the post-action world state.
- **`--fast`** — deterministic step (no wall-clock sleep); runs as fast as the CPU allows
  (`--fast-steps N` sets physics ticks per env step).

Monitoring: TensorBoard event files are written under `logs/`. Optional Weights & Biases
mirroring is available — set `use_wandb: True` in `config/vx300s_reacher_sac.yaml` (requires
`pip install wandb` + `wandb login`).

## Notes / known first-run checks
- `control_period` (`config/vx300s_mujoco_plugins.yaml`) should be >= the MJCF `<option timestep>`;
  the plugin warns otherwise.
- If the arm droops or oscillates under torque tracking, tune the controller gains in
  `config/vx300s_mujoco_control.yaml`.
- Standard reach only for now; the goal-conditioned (SAC/TD3 + HER) variant and pnp/push tasks
  are follow-ups.
