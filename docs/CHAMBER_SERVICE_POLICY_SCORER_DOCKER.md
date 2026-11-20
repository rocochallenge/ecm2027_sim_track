# Chamber Service：策略与评测容器解耦

可以把 Isaac/PhysX 环境和评分放在环境容器，把策略放在独立的 policy 容器。环境容器只持有任务、传感器、成功判定和评分状态；policy 容器只接收 observation 并返回 action。这样策略镜像不需要安装 Isaac Sim，也不能绕过环境里的 success/timeout/score。

推荐链路如下：

```text
policy Docker --HTTP/IPC--> env Docker (Isaac Sim + ChamberServiceTask)
       observation <-------- reset/step -------- action
```

仓库已有 `robofinals/scripts/env_server.py` 和 REST/IPC wrapper。REST wrapper 现在的协议是：

```http
POST /attach
POST /reset  {"session_id": "..."}
POST /step   {"session_id": "...", "action": [0.0, ...]}
GET  /metrics
```

`/reset` 和 `/step` 返回两个观测字段：

* `observation`：结构化 observation 的 JSON 形式，供 policy 使用；
* `obs`：兼容旧客户端的 base64 图像列表。

`action` 同时支持 JSON 数组和旧的空格分隔字符串。数组长度必须等于环境的 action dimension；Galbot-Mobile 当前底盘动作是 `[vx, vy, wz]`，拼在上肢动作之后。

评测结束后应由 evaluator（评测控制端）调用 `/metrics` 读取环境容器维护的 `check_success` 和 `chamber_process.process_score`；不要把此端点开放给 policy。该端点会返回成功状态、逐项分数和阶段信息，属于评分侧信息。

> **当前 REST 实现的安全边界：** 服务默认监听 `0.0.0.0:8000` 且没有鉴权。`/step` 返回 `reward` 和 `info`，`/metrics` 返回成功状态、逐项分数和阶段信息，`/shutdown` 可停止服务。因此不要将该 REST 服务直接暴露给不可信 policy 或外部网络。若要使用容器隔离，需由 evaluator 加认证和路由访问控制，只向 policy 代理最小化的 observation/action 接口，并仅允许 evaluator 读取评分端点。
>
> 仓库内 IPC `RemoteEnv.step()` 会丢弃 reward 和 info，只向策略返回 observation、terminated 和 truncated；IPC 方法仍包含 `close`，并使用默认共享密钥 `lightwheel`，因此它是受限协议，不是恶意代码沙箱，应只在受控本机网络中使用。

环境容器示例：

```bash
docker run --rm --gpus all --ipc=host --network=host \
  -e DISPLAY="$DISPLAY" -e NVIDIA_VISIBLE_DEVICES=all \
  harbor.lightwheel.net/kun_delivery/robofinals:<tag> \
  /opt/conda/envs/robofinals/bin/python \
  /workspace/robofinals/robofinals/scripts/env_server.py \
  --remote_protocol=restful --restful_host=0.0.0.0 --restful_port=8000
```

policy 容器只需要 HTTP client、模型依赖和权重：

```python
import requests

base = "http://env-host:8000"
session_id = requests.post(f"{base}/attach", json={}).json()["session_id"]
obs = requests.post(f"{base}/reset", json={"session_id": session_id}).json()["observation"]
while True:
    action = policy(obs)
    result = requests.post(
        f"{base}/step",
        json={"session_id": session_id, "action": action},
    ).json()
    obs = result["observation"]
    if any(result.get(key) for key in ("done", "truncated")):
        break
```

实际云评测时建议：

1. 为环境容器分配 GPU 和 Isaac 运行时；policy 容器只分配模型所需 GPU/CPU。
2. 使用 `--network=host` 或同一 Docker network；生产部署增加鉴权、请求超时和版本化 observation/action schema。
3. 由环境容器统一处理 900 秒仿真超时、`check_success` 和 `chamber_process.process_score`；由 evaluator 单独读取评分结果，policy 只接收获准的 observation 并返回 action。
4. 先用单环境、单请求串行模式验证 action dimension 和相机字段，再考虑批量环境或 gRPC/共享内存。REST 的 JSON 会复制图像和 tensor，吞吐要求高时切换 IPC 或二进制协议。

因此新增接口的代码量很小；真正需要单独规划的是两个镜像的网络、schema 版本、鉴权和高吞吐传输方式。
