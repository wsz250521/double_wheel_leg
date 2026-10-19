# Wheel Legged Gym 环境 #

### 致谢

Wheel-Legged-Gym 的实现依赖于来自 [legged_gym](https://github.com/leggedrobotics/legged_gym) 和 [rsl_rl](https://github.com/leggedrobotics/rsl_rl) 项目的资源,这两个项目由 [Robotic Systems Lab](https://rsl.ethz.ch/) 创建。

相关链接:

- [Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning](https://arxiv.org/abs/2109.11971)(使用大规模并行深度强化学习在数分钟内学会行走)
- [Concurrent Training of a Control Policy and a State Estimator for Dynamic and Robust Legged Locomotion](https://arxiv.org/abs/2202.05481)(针对动态且鲁棒的腿式运动,同时训练控制策略与状态估计器)

### 安装 ###
1. 使用 Python 3.6、3.7 或 3.8(推荐 3.8)创建一个新的 Python 虚拟环境
2. 从 https://pytorch.org/get-started/ 安装支持 CUDA 的 PyTorch
3. 安装 Isaac Gym
   - 从 https://developer.nvidia.com/isaac-gym 下载并安装 Isaac Gym Preview 4
   - `cd isaacgym/python && pip install -e .`
   - 尝试运行示例:`cd examples && python 1080_balls_of_solitude.py`
   - 如需排错,请参阅文档 `isaacgym/docs/index.html`
4. 安装 wheel_legged_gym
    - 克隆本仓库
   - `cd Wheel-Legged-Gym && pip install -e .`

### 代码结构 ###
1. 每个环境由环境类文件(`legged_robot.py`)和配置文件(`legged_robot_config.py`)共同定义。配置文件中包含两个类:一个包含所有环境参数(`LeggedRobotCfg`),另一个包含训练参数(`LeggedRobotCfgPPo`)。
2. 环境类与配置类都使用继承机制。
3. 在 `cfg` 中指定的每一个非零奖励权重,都会向奖励元素列表中添加一个同名函数,所有奖励函数求和后得到总奖励。
4. 任务必须使用 `task_registry.register(name, EnvClass, EnvConfig, TrainConfig)` 进行注册。注册过程在 `envs/__init__.py` 中完成,但也可以在仓库外部进行注册。

### 使用方法 ###
1. 训练:
    ```python wheel_legged_gym/scripts/train.py --task=wheel_legged_vmc_flat```
    - 若要在 CPU 上运行,请添加以下参数:`--sim_device=cpu`、`--rl_device=cpu`(仿真在 CPU、RL 在 GPU 是可行的)。
    - 若要无界面运行(不进行渲染),请添加 `--headless`。
    - **重要提示**:为了提升性能,训练一旦启动,请按 `v` 键停止渲染。之后可以随时重新开启以检查训练进度。
    - 训练好的策略会保存在 `logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`。其中 `<experiment_name>` 与 `<run_name>` 在训练配置中定义。
    - 使用 TensorBoard 监控训练过程:`tensorboard --logdir=./ --port=8080`
    - 以下命令行参数会覆盖配置文件中设置的对应值:
     - `--task TASK`:任务名称。
     - `--resume`:从检查点恢复训练。
     - `--experiment_name EXPERIMENT_NAME`:要运行或加载的实验名称。
     - `--run_name RUN_NAME`:本次运行的名称。
     - `--load_run LOAD_RUN`:恢复训练时要加载的运行名称。设为 -1 时将加载最近一次运行。
     - `--checkpoint CHECKPOINT`:保存的模型检查点编号。设为 -1 时将加载最新的检查点。
     - `--num_envs NUM_ENVS`:要创建的环境数量。
     - `--seed SEED`:随机种子。
     - `--max_iterations MAX_ITERATIONS`:最大训练迭代次数。
     - `--exptid EXPTID`:实验编号。
2. 运行训练好的策略:
   ```python wheel_legged_gym/scripts/play.py --task=wheel_legged_vmc_flat```
    - 默认情况下,加载的是该实验文件夹下最近一次运行中的最后一个模型。
    - 也可以通过在训练配置中设置 `load_run` 与 `checkpoint` 来选择其他运行/模型迭代次数。
3. 已有的任务:
   - `wheel_legged`:针对开链机器人在多种地形上的端到端训练。
   - `wheel_legged_vmc`:使用 VMC 统一开链与闭链机构的运动控制,便于将策略部署到闭链机器人上。
   - `wheel_legged_vmc_flat`:在平坦地形上训练机器人(显存需求较低)。

### 添加新的环境 ###
基础环境 `legged_robot` 实现了一个复杂地形下的运动任务。对应的 cfg 并没有指定机器人资源(URDF/MJCF),也没有设置奖励权重。

1. 在 `envs/` 目录下添加一个新文件夹,并在其中创建继承自现有环境 cfg 的 `'<your_env>_config.py`。
2. 如果要添加新的机器人:
    - 将相应的资源文件放入 `resources/` 目录。
    - 在 `cfg` 中设置资源路径,定义机体名称、默认关节位置(`default_joint_positions`)以及 PD 增益。指定期望的 `train_cfg` 以及环境(类)的名称。
    - 在 `train_cfg` 中设置 `experiment_name` 与 `run_name`。
3. (如需要) 在 `<your_env>.py` 中实现你的环境,继承自现有环境,重写需要的方法或添加你自己的奖励函数。
4. 在 `envs/__init__.py` 中注册你的环境。
5. 根据需要修改/调整 `cfg` 与 `cfg_train` 中的其他参数。若要去除某项奖励,将其权重设为 0 即可。请勿修改其他环境的参数!

### 故障排查 ###
1. 如果出现如下错误:`ImportError: libpython3.8m.so.1.0: cannot open shared object file: No such file or directory`,请执行:`sudo apt install libpython3.8`。如果仍不生效,可能还需要执行 `export LD_LIBRARY_PATH=/path/to/libpython/directory`,或者(对于 conda 用户)`export LD_LIBRARY_PATH=/path/to/conda/envs/your_env/lib`(请将 `/path/to/` 替换为对应的实际路径)。

### 已知问题 ###
1. 在 GPU 上使用三角网格地形进行仿真时,由 `net_contact_force_tensor` 报告的接触力并不可靠。一种可行的解决办法是使用力传感器,但此时力会通过相邻机体的传感器传播,从而产生不期望的行为。然而,对于腿式机器人,可以只在足端/末端执行器上添加传感器,这样能得到预期的结果。使用力传感器时,请确保通过 `sensor_options.enable_forward_dynamics_forces` 将重力从报告的力中剔除。示例:
```
    sensor_pose = gymapi.Transform()
    for name in feet_names:
        sensor_options = gymapi.ForceSensorProperties()
        sensor_options.enable_forward_dynamics_forces = False  # 例如排除重力
        sensor_options.enable_constraint_solver_forces = True  # 例如保留接触力
        sensor_options.use_world_frame = True  # 以世界坐标系报告力(便于获取垂直分量)
        index = self.gym.find_asset_rigid_body_index(robot_asset, name)
        self.gym.create_asset_force_sensor(robot_asset, index, sensor_pose, sensor_options)
    (...)

    sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
    self.gym.refresh_force_sensor_tensor(self.sim)
    force_sensor_readings = gymtorch.wrap_tensor(sensor_tensor)
    self.sensor_forces = force_sensor_readings.view(self.num_envs, 4, 6)[..., :3]
    (...)

    self.gym.refresh_force_sensor_tensor(self.sim)
    contact = self.sensor_forces[:, :, 2] > 1.
```