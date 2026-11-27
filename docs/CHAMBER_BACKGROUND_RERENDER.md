# 已有轨迹更换背景并重新生成图像

使用 state replay，把记录的机器人及物体状态恢复到新 USD 场景中，重新渲染相机。
适用于只修改背景、材质等视觉内容。若改变桌面、任务物体位置或碰撞几何，需另行确认原轨迹是否仍适用。

## ChamberService 运行命令

```bash
# 在本仓库根目录执行；先激活安装本仓库依赖的 Python 环境。
./scripts/run_rerender.sh \
  --dataset_file /absolute/path/to/galbot_success.hdf5 \
  --layout "$PWD/robofinals/data/assets/SceneWithBackgroud/scene04_1.usd" \
  --enable_full_local_scene
```

脚本复用 `scripts/runtime_common.sh`，自动将本仓库和 `vendor/isaaclab_arena` 加入导入路径，
优先使用 `.venv/bin/python`，否则使用当前环境的 `python`。也可以显式指定解释器：

```bash
PYTHON_BIN=/path/to/env/bin/python ./scripts/run_rerender.sh \
  --dataset_file /absolute/path/to/galbot_success.hdf5 \
  --layout "$PWD/robofinals/data/assets/SceneWithBackgroud/scene04_1.usd" \
  --enable_full_local_scene
```

- `--dataset_file`：输入轨迹文件，保持不变。
- `--layout`：新场景的源 USD。使用 `scene04_1.usd`，不要把预处理生成的 `_enabled.usd` 再当源文件输入。
- `--enable_full_local_scene`：与当前 Galbot 遥操的完整本地场景加载配置一致。
- `--record --enable_cameras`：同时生成相机数据和 MP4。
- `--device cpu`：使用 CPU 物理状态；RTX 相机渲染仍使用 NVIDIA GPU。

路径需根据实际 checkout、环境和数据目录调整。场景 USD 及其引用的资产、材质和贴图必须在本机可访问。

## 输出及覆盖范围

在输入 HDF5 的同目录生成：

- `dataset_with_video.hdf5`：覆盖已有同名文件。
- 时间戳命名的 `.mp4`：相机横向拼接视频，保留以前的视频，请按文件名选择最新一次输出。

HDF5 中的“视频”是 `data/demo_N/obs/*_camera_rgb` 图像数组，不是内嵌 MP4。
这些数组会被新渲染图像替换，原 actions、states 和其他数据保持不变。
输出保留根属性（包括四元数格式版本），并更新 `env_args` 中的场景路径。

**当前脚本只处理编号最大的一个 episode。** 其他 episode 会原样复制；若文件含多个 episode，不能把这次输出当作全文件已完成背景刷新。
ChamberService 示例输入只有 `demo_0`。
不要将输出文件本身作为输入；重复重刷应继续使用原始轨迹。

## 内存和录制实现

state replay 只读取 `initial_state` 和 `states`，跳过旧 RGB。
示例原文件旧 observations 解压约 5.14 GiB，而所需状态约 4.16 MiB。
新图像逐帧写入 `.camera_frames.h5` 临时文件，完成后分块写入输出 HDF5，再删除临时文件。
这降低了数据加载和相机缓存的内存开销，但 Isaac Sim、完整场景和 RTX 仍需要内存及显存。
磁盘需要同时容纳输入、临时图像文件和输出。

录制跳过额外的 initial_state reset，每个轨迹状态只录一帧；headless 下仍显式渲染并刷新相机。
ChamberService + Galbot 完整场景回放会隐藏场景内重复的 `FlatGrid/Environment/Geometry` 地面网格，
保留 Galbot 的蓝色地面和地面碰撞，避免共面地面争用深度。
输出前校验相机帧数；没有图像或帧数不完整时会报错，不开始覆盖输出 HDF5。

## 完成和验证

正常日志包含：

```text
[replay_local] loaded trajectory only: initial_state, states (old RGB skipped)
Dataset with video saved to: .../dataset_with_video.hdf5
Save combined video completed, 4503 frames
```

4503 是该示例轨迹的帧数，其他数据应以自己的 `num_samples` 为准。
MP4 当前固定按 30 fps 编码，4503 帧对应 150.1 秒；这不代表原轨迹的仿真时长。

```bash
python - <<'PY'
import h5py

path = '/absolute/path/to/dataset_with_video.hdf5'
with h5py.File(path, 'r') as f:
    demo = f['data/demo_0']
    expected = int(demo.attrs['num_samples'])
    for name in ('first_person_camera_rgb', 'left_hand_camera_rgb', 'right_hand_camera_rgb'):
        images = demo['obs'][name]
        print(name, images.shape)
        assert len(images) == expected
PY
```

还需播放最新 MP4，检查开头、中段、结尾的运动及背景材质。
帧数相同不能单独证明渲染正确。若日志出现贴图无法解析，需检查实际加载的 USD 和贴图路径；不能仅根据环境启动成功认定材质完整。

## 本次验证记录

2026-11-26：在原始完整文件上验证选择性加载，状态张量与文件逐项一致且不含旧 observations。
使用完整场景、CPU 和相机完成 30 个采样状态的真实回放，生成 HDF5 和 MP4；三路图像均为 30 帧且已刷新，非相机数据及 `format_version` 保持一致。
这次输入内存优化后未重新验证完整 4503 帧运行。

移植来源：主仓库提交 `c0bede8ed`。本缩减仓库补充 `scripts/run_rerender.sh` 并使用 vendor Arena 路径。

同日移植验证：在缩减仓库通过 `run_rerender.sh`、完整场景、CPU 和相机完成 30 帧真实回放，
三路 RGB 各 30 帧、非相机数据及根属性保持一致，MP4 正常生成。
首尾抽帧可见机器人运动和蓝色网格地面；未在该分支重跑完整 4503 帧。
