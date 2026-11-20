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

当前满分（与 `ChamberServiceTask.MAX_SCORE` 和 `POINT_NAMES` 对齐）：

```text
阶段 1 打开腔体 2 分 + 阶段 2 取出原工件 1 分 + 阶段 3 将原工件放入盒中 1 分
+ 阶段 4 拿起海绵并擦拭 5 个腔体面（1 + 5）6 分
+ 阶段 5 取出并放置替换工件 2 分 + 阶段 6 关闭腔体 1 分 = 13 分
```

每个计分点每轮最多计一次；后续阶段只会在前一阶段的所有计分点完成后启用。5 个擦拭面分别是 x-/x+/y-/y+ 内壁和底面。

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
