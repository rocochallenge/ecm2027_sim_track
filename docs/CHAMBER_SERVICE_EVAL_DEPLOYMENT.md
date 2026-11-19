# Chamber Service 客户评测部署说明

本文档描述交付给客户的 Chamber Service 在线评测链路。评测由两个进程组成：

```text
policy 进程 / policy Docker
        │ 受限 IPC：attach、reset、step、get_metadata、close
        ▼
仿真进程 / simulator Docker
        ├─ Isaac Sim / PhysX
        ├─ ChamberServiceTask
        ├─ check_success
        ├─ 900 秒仿真时间超时
        └─ chamber_process.process_score
```

policy 容器不需要安装 Isaac Sim，也不应挂载仿真容器的源码目录。仿真容器是评分的唯一权威；policy 只能根据 observation 产生 action。

## 1. 启动仿真进程

仿真进程使用 `env_server.py`，默认协议是 IPC，默认地址为 `0.0.0.0:50000`，认证 key 默认是 `lightwheel`。

宿主机或仿真容器中启动：

```bash
docker run --rm -it \
  --gpus all --ipc=host --network=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e DISPLAY="${DISPLAY:-:0}" \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$PWD/eval_result:/workspace/eval_result" \
  harbor.lightwheel.net/kun_delivery/robofinals:<tag> \
  /opt/conda/envs/robofinals/bin/python \
  /workspace/robofinals/robofinals/scripts/env_server.py \
  --remote_protocol=ipc \
  --ipc_host=0.0.0.0 \
  --ipc_port=50000 \
  --ipc_authkey=lightwheel \
  --eval_metrics_path=/workspace/eval_result/chamber_service_metrics.json
```

服务器启动后会等待 policy 客户端连接。Isaac Sim、任务 reset、contact sensor、success 和评分都在这个进程中执行。

## 2. 启动 policy 进程

policy 进程使用对应的部署配置：

```bash
python robofinals/scripts/eval/policy.py \
  --config policy/PI/deploy_policy_wrc_demo_task.yml \
  --overrides \
  --env_cfg:task ChamberServiceTask \
  --env_cfg:robot Galbot-Mobile \
  --env_cfg:layout robofinals/data/assets/Scene/scene04_1.usd \
  --env_cfg:scene_backend local \
  --env_cfg:task_backend local \
  --env_cfg:enable_full_local_scene True \
  --test_num 1
```

policy 进程连接 `127.0.0.1:50000`。如果 policy 在另一个容器中，两个容器必须共享网络，或者把 policy 代码中的 IPC 地址改成仿真容器的可达地址；认证 key 必须一致。

policy 进程完成所有 episode 后，停止仿真容器（Ctrl-C 或容器停止）以触发最终 metrics JSON 写盘。

当前 policy 入口不再使用 `scene.get_state`、`reset_to` 或通用远程属性代理。`eval_initial_states_path` 也不属于受限协议；如果客户需要固定初始状态，应由仿真侧实现 episode/reset 配置。

## 3. IPC 接口

IPC 使用 Python `multiprocessing.managers.BaseManager`，但 policy 端只拿到以下五个白名单方法：

| 方法 | policy -> 仿真 | 返回内容 |
|---|---|---|
| `attach(env_cfg)` | 任务、机器人、场景和运行参数 | `actions_dim`、`decimation`、`num_envs` |
| `reset(seed=None)` | 可选 seed | `(observation, reset_info)` |
| `step(action)` | 一个 action tensor/数组 | `(observation, terminated, truncated)` |
| `get_metadata()` | 无参数 | action 维度、控制 decimation、环境数量 |
| `close()` | 无参数 | 关闭仿真环境 |

### policy -> 仿真：attach

`env_cfg` 是启动环境所需的配置，不是仿真状态。客户应把 task、robot、layout、backend、camera 和 seed 写进配置文件或启动参数。

### policy -> 仿真：action

`action` 是 shape 为 `(num_envs, actions_dim)` 的浮点数组。Galbot-Mobile 当前 action 由上肢 action 加底盘 `[vx, vy, wz]` 组成；具体维度由 `get_metadata()` 返回，policy 不应硬编码。

### 仿真 -> policy：observation

`observation` 是环境 policy observation group 的原始结构，通常包括：

- 机器人关节位置/速度等本体状态；
- 双臂末端位姿和夹爪状态；
- 相机 observation（如果启动时启用 camera）；
- 任务定义的公开 instruction 或 observation term。

`observation` 不包含：

- fixture 或物体的内部 root state；
- contact sensor 原始结果；
- `_stage_flags`、擦拭面计数和 `process_score`；
- `check_success` 内部状态；
- 仿真 USD stage、scene 对象或 task 配置对象。

`terminated`/`truncated` 只是 episode 控制信号，用于让 policy 停止当前 rollout。reward、info、success 标志和内部评分不会通过 policy-facing `step()` 返回。

## 4. 评分在哪里计算

评分完全在仿真进程中计算。仿真进程退出时会把最终结果写入
`--eval_metrics_path` 指定的 JSON 文件，默认是
`./eval_result/chamber_service_metrics.json`：

- 文件：[chamber_service_task.py](/home/kun/lw_benchhub/robofinals_tasks/chamber_service_task.py)
- `_check_success()`：原有阶段 success 判定；评分不反向参与 success。
- `_complete_stage()`：普通 stage 完成时累计 5 分。
- `_update_sponge_chamber_contacts()`：擦拭每次合格接触累计 1 分，共 5 个面。
- `get_process_score_metrics()`：输出当前分数、每个环境分数、擦拭面数和最高分。
- `get_checker_results()`：把结果放进 `chamber_process`。

当前满分：

```text
5 个普通 stage × 5 分 + 5 个擦拭面 × 1 分 = 30 分
```

Chamber Service 的仿真超时为 900 秒，使用仿真时间，不依赖 wall clock。客户部署时应以这个仿真侧 JSON 为准，不以 policy 进程打印的 success 结果为准。建议把 `/workspace/eval_result` 挂载到宿主机。

## 5. 客户如何修改评分规则

只需要修改 task 文件，不要修改 policy 接口：

1. 修改普通 stage 分值：调整 `STAGE_SCORE`，或在 `_complete_stage()` 中为不同 `stage_idx` 建立分值表。
2. 修改擦拭面数量：调整 `SPONGE_CHAMBER_CONTACTS_REQUIRED`。
3. 修改擦拭一次的有效条件：调整 `_update_sponge_chamber_contacts()` 中的接触持续时间、接触状态和 stage 前置条件。
4. 修改最终评分输出：调整 `get_process_score_metrics()`，并保持 `get_checker_results()` 返回 `chamber_process`。
5. 修改 success 规则：只改 `_check_success()` 中的 stage flag 条件；不要用 score 反向 gate success，除非评测协议明确要求这样做。

修改后至少执行：

```bash
python -m py_compile robofinals_tasks/chamber_service_task.py
git diff --check
```

然后用真实 Isaac 启动一次 Chamber Service eval，确认 reset、step、15 分钟超时、success 和最终 `chamber_process` 都能正常产生。静态检查不能替代 Isaac Sim 运行验证。

## 6. 部署边界

- policy 镜像只部署模型和客户端，不安装或挂载 Isaac Sim 内部资产；
- 仿真镜像不加载客户 policy 代码；
- IPC 端口只允许 policy 容器访问；
- 不要把 `/metrics`、日志目录或仿真内部状态端口暴露给 policy；
- 客户自己的评测调度器应负责收集仿真侧 score、success、timeout 和 episode 编号。

这样 policy 的职责只有 `observation -> action`，评分和任务判定始终由仿真侧掌握。

### Galbot-Mobile：兼容 19 / 20 维 action

仿真入口按每次 action 的最后一维自动识别格式，policy 不需要切换机器人配置：

| 输入 shape | 最后几维 | 含义 |
|---|---|---|
| `(num_envs, 19)` | `action[:, 16:19]` | 机体坐标系 `[vx, vy, wz]`，单位 m/s、m/s、rad/s；保留现有平滑 |
| `(num_envs, 20)` | `action[:, 16:20]` | 旧遥操录制的 `[w1, w2, w3, w4]`，按下面的映射恢复底盘速度；不重复平滑 |

两种格式前 16 维保持一致。支持 torch tensor、NumPy 数组及浮点列表。
20 维使用数据集原始 action 值，不是乘过 `wheel_scale=0.01` 的执行器速度，也不是任意真机轮子的 rad/s：

```text
vx = ( w1 + w2 + w3 + w4) / 4
vy = (-w1 + w2 + w3 - w4) / 4
wz = (-w1 + w2 - w3 + w4) / 4
```

四个值都为 0.4 时恢复 `[0.4, 0, 0]`；横移对应 `[-v, v, v, -v]`；转向对应 `[-w, w, -w, w]`。
不能对这组值再乘遥操摇杆的横向比例 0.7。
任意四轮值中不属于平面运动的分量会被上述映射投影掉。

两种输入都进入同一个 `GalbotBaseVelocityAction.apply_actions()`，由它设置轮子目标速度和底盘 root velocity。
内部 action space 和 observation 中的 `actions` 仍为 19 维。
IPC metadata 保留 `actions_dim: 19` 表示内部默认维度，另返回 `accepted_action_dims: [19, 20]` 和 `action_formats`。
输出 20 维的现有模型应使用自己的 20 维输出头，不要根据默认 `actions_dim` 把它裁成 19 维。

旧 20 维 HDF5 可以直接使用 `scripts/run_replay_eval.sh ... action`，无需改写原数据文件。
这只兼容动作接口，不保证不同资产、物理参数、初始状态下的历史回放轨迹完全一致。

### Policy 进程退出与超时

评测入口会在正常完成、policy 异常、键盘中断和 watchdog 超时这几种路径统一关闭 IPC 环境和 video writer，避免 policy 进程或仿真连接残留。ChamberServiceTask 仍以 900 秒仿真时间作为 episode 截止条件；policy 入口另外提供 wall-clock safety watchdog，默认 900 秒，防止 policy 自己不再调用 `env.step()` 时永久挂起。可在 policy YAML 中调整或关闭：

```yaml
policy_process_timeout_s: 900
```

设为 `0` 表示关闭 wall-clock watchdog。客户 policy 应在收到 `terminated` 或 `truncated` 后停止当前 episode；入口的 `finally` 清理保证即使 policy 抛异常也会断开 IPC 并退出。
