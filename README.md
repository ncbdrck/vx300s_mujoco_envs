# vx300s_mujoco_envs

> ⚠️ **Under development.** Part of the experimental MuJoCo backend for MultiROS/UniROS
> (`multiros` branch `feature/mujoco-backend`). APIs and structure may change; not yet merged
> into the stable release.

ViperX-300 S MuJoCo environments (`mujoco_ros_pkgs` backend), used to validate the MultiROS
MuJoCo backend end-to-end. The package mirrors the existing Gazebo VX300S envs but builds each
env entirely from the `multiros` MuJoCo tooling (`mujoco_core` / `mujoco_models` / `mujoco_physics`
/ `MujocoBaseEnv`), so creating a MuJoCo env follows the same workflow as a Gazebo one.

Currently implemented: **reach**, **push**. Planned: **pick-and-place**, and goal-conditioned
(HER) variants.

The arm is driven purely through the `ros_control` trajectory interface
(`/vx300s/arm_controller/command`); **MoveIt is not used** — end-effector poses come from forward
kinematics, joint state from `/vx300s/joint_states`.

It reuses the robot description `viperx300s_description` (`vx300s.urdf.xacro`); the MuJoCo backend
strips the gripper/finger transmissions automatically (via `controlled_joints`) since they have no
joints in the MJCF. Task and training configs are self-contained in this package's `config/`.

## Tasks

| Task | Env id | Scene | Scripts |
|------|--------|-------|---------|
| Reach | `VX300SMujocoReacherSim-v0` | `vx300s_table_scene.xml` (arm + table) | `vx300s_mujoco_reach_{test,train,validate}.py` |
| Push  | `VX300SMujocoPushSim-v0`    | `vx300s_push_scene.xml` (table + cube) | `vx300s_mujoco_push_{test,train,validate,poke}.py` |

- **Reach** — move the end-effector to a sampled goal (RViz marker; no runtime objects).
- **Push** — slide a cube resting on the table to a goal region. The cube is a free-joint body in
  the MJCF, repositioned each episode via `mujoco_set_body_state` and read back via
  `mujoco_get_body_state` (no per-episode spawn/delete).

## Layout
- `assets/vx300s_mjcf/` — the Trossen VX300S model from MuJoCo Menagerie, with its native MJCF
  actuators removed (joints are driven by `mujoco_ros_control`). Scenes:
  - `vx300s.xml` — arm only.
  - `vx300s_table_scene.xml` — arm + table (top face at z = 0) + ground plane (shared by all tasks).
  - `vx300s_push_scene.xml` — table scene + a free-joint cube.
  - `vx300s_scene.xml` — arm + ground plane only (original reach scene, kept for reference).
- `config/vx300s_mujoco_control.yaml` — ros_control controllers (effort JointTrajectoryController).
- `config/vx300s_mujoco_plugins.yaml` — the `MujocoPlugins` block that loads the ros_control bridge.
- `config/initial_joint_states.yaml` — home configuration applied on load/reset.
- `config/vx300s_reach_task_config.yaml`, `config/vx300s_push_task_config.yaml` — task params
  (reward, workspace, goal sampling; push adds cube spawn / goal / observation regions).
- `config/vx300s_reacher_sac.yaml`, `vx300s_reacher_td3.yaml`, `vx300s_pusher_sac.yaml` — SB3
  training hyper-parameters.
- `launch/vx300s_mujoco_reach.launch`, `vx300s_mujoco_push.launch` — bring up the URDF, the
  mujoco_ros server (with the right scene) and the controllers.
- `src/vx300s_mujoco_envs/robot_envs/` — the shared VX300S robot env.
- `src/vx300s_mujoco_envs/task_envs/reach/`, `task_envs/push/` — the task envs.

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
everything:
```bash
rosrun vx300s_mujoco_envs vx300s_mujoco_reach_test.py   # reach
rosrun vx300s_mujoco_envs vx300s_mujoco_push_test.py    # push
```
Equivalently, in Python:
```python
import uniros as gym
env = gym.make("VX300SMujocoPushSim-v0")   # launch_mujoco / new_roscore / load_robot default True
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

Terminal 1 — sim + controllers (use the launch file matching the task):
```bash
roslaunch vx300s_mujoco_envs vx300s_mujoco_push.launch gui:=true
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
# Reach
rosrun vx300s_mujoco_envs vx300s_mujoco_reach_train.py
rosrun vx300s_mujoco_envs vx300s_mujoco_reach_validate.py --episodes 20

# Push
rosrun vx300s_mujoco_envs vx300s_mujoco_push_train.py
rosrun vx300s_mujoco_envs vx300s_mujoco_push_validate.py --episodes 20 --model-tag trained_model_push
```

Common training flags: `--no-realtime` (paused MDP loop), `--fast` (deterministic fast-step,
`--fast-steps N`), `--steps N` (override the config step count), `--attach`, `--mujoco-gui`.

Training regimes:
- **real-time** (default) — UniROS paper §7 loop; physics never pauses (sim→real fidelity).
- **`--no-realtime`** — paused MDP loop; every sample is the post-action world state.
- **`--fast`** — deterministic step (no wall-clock sleep); runs as fast as the CPU allows.

### Contact check (push)

Random joint actions rarely bring the end-effector to the cube, so they do not exercise contact.
The poke script drives the end-effector through the cube and reports how far it moved:
```bash
rosrun vx300s_mujoco_envs vx300s_mujoco_push_poke.py --attach
```

### Monitoring

TensorBoard event files are written under `logs/`:
```bash
tensorboard --logdir $(rospack find vx300s_mujoco_envs)/logs
```
Optional Weights & Biases mirroring: set `use_wandb: True` in the training config
(`config/vx300s_reacher_sac.yaml` / `vx300s_pusher_sac.yaml`) and run `pip install wandb` +
`wandb login`. The run mirrors the same metrics SB3 writes to TensorBoard.

## Notes / known first-run checks
- `control_period` (`config/vx300s_mujoco_plugins.yaml`) should be >= the MJCF `<option timestep>`;
  the plugin warns otherwise.
- If the arm droops or oscillates under torque tracking, tune the controller gains in
  `config/vx300s_mujoco_control.yaml`.
- **Push contact / safety floor.** The per-link FK safety check rejects actions that would dip a
  link below `table_z + safety_z_margin`. The defaults suit reach; if the arm cannot get low enough
  to contact the cube, tune `table_z` / `safety_z_margin` in `config/vx300s_push_task_config.yaml`.
- If the cube slides or tips unrealistically, tune the cube `friction` (and `solref`/`solimp`) in
  `assets/vx300s_mjcf/vx300s_push_scene.xml`.
