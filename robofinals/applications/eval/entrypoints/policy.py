# Copyright 2025 Lightwheel Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to replay demonstrations with Isaac Lab environments."""

"""Launch Isaac Sim Simulator first."""

import argparse
import contextlib
import importlib
import json
import os
import sys
import numpy as np
import tqdm
import yaml

from pathlib import Path

sys.path.append("./")
sys.path.append("../../policy")


def _to_numpy_bool(value):
    """Normalize a policy eval result (bool / np / torch) to a numpy bool array."""
    try:
        import torch

        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(value).astype(bool)


def _expand_path(value):
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    return value


def _prepare_config(config):
    if "checkpoint" in config:
        config["checkpoint"] = _expand_path(config["checkpoint"])
    if "eval_initial_states_path" in config:
        config["eval_initial_states_path"] = _expand_path(config["eval_initial_states_path"])
    env_cfg = config.get("env_cfg")
    if isinstance(env_cfg, dict) and "layout" in env_cfg:
        env_cfg["layout"] = _expand_path(env_cfg["layout"])
    return config


# add argparse arguments
# Get project root directory (assuming script is in robofinals/scripts/policy/)
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent.parent.parent
default_config_path = project_root / "policy" / "PI" / "deploy_policy_lerobot.yml"

parser = argparse.ArgumentParser(description="Eval policy in Isaac Lab environments.")
parser.add_argument("--config", type=str, default=str(default_config_path))
parser.add_argument("--overrides", nargs=argparse.REMAINDER)
parser.add_argument("--save_states", action="store_true", help="Save states after each step")

# parse the arguments
args_cli = parser.parse_args()


def parse_args_and_config():

    with open(args_cli.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]

            try:
                value = eval(value)
            except Exception:
                pass

            # use ':' to split config
            if ':' in key:
                keys = key.split(':')
                current_level = override_dict
                for k in keys[:-1]:
                    if k not in current_level:
                        current_level[k] = {}
                    current_level = current_level[k]
                current_level[keys[-1]] = value
            else:
                override_dict[key] = value

        return override_dict

    def deep_merge(original, update):
        for key, value in update.items():
            if (key in original and
                isinstance(original[key], dict) and
                    isinstance(value, dict)):
                deep_merge(original[key], value)
            else:
                original[key] = value
        return original

    if args_cli.overrides:
        overrides = parse_override_pairs(args_cli.overrides)
        config = deep_merge(config, overrides)

    return _prepare_config(config)


def main(usr_args):

    from robofinals.distributed.proxy import RemoteEnv
    env = RemoteEnv.make(address=('127.0.0.1', 50000), authkey=b'lightwheel')
    from robofinals.distributed.restful import DotDict
    if "env_cfg" in usr_args and usr_args["env_cfg"]:
        env_cfg = DotDict(usr_args["env_cfg"])
        defaults = {
            "scene_backend": "robocasa",
            "task_backend": "robocasa",
            "device": "cuda:0",
            "robot_scale": 1.0,
            "first_person_view": False,
            "disable_fabric": False,
            "num_envs": 1,
            "usd_simplify": False,
            "video": False,
            "for_rl": False,
            "variant": "Visual",
            "concatenate_terms": False,
            "distributed": False,
            "physics_backend": None,
            "seed": 42,
            "sources": None,
            "object_projects": None,
            "execute_mode": "eval",
            "replay_cfgs": {"add_camera_to_observation": True},
            "enable_full_local_scene": False,
        }
        for key, value in defaults.items():
            if key not in env_cfg:
                env_cfg[key] = value
    policy_name = usr_args["policy_name"]
    module_name, separator, class_name = policy_name.partition(":")
    if not separator:
        raise ValueError("policy_name must be an import path such as my_policy:MyPolicy")
    policy_module = importlib.import_module(module_name)
    policy_class = getattr(policy_module, class_name)
    policy = policy_class(usr_args)

    eval_meta = env.attach(env_cfg)
    usr_args['actions_dim'] = eval_meta['actions_dim']
    usr_args['decimation'] = eval_meta['decimation']

    eval_initial_states_path = usr_args.get("eval_initial_states_path", None)
    if eval_initial_states_path:
        raise ValueError(
            "eval_initial_states_path is not supported by the restricted policy protocol; "
            "initial states must be selected by the simulator-side evaluator."
        )

    ori_test_num = usr_args.get('test_num', 10)  # default 10
    test_num = ori_test_num

    # Each env runs an independent episode, so one batched rollout evaluates
    # ``num_envs`` episodes at once. Run enough batches to cover ``test_num``.
    num_envs = max(1, int(usr_args.get("env_cfg", {}).get("num_envs", 1)))
    num_batches = (test_num + num_envs - 1) // num_envs

    # Optional: record the initial pose of selected objects for each episode,
    # written in the put_bottles.json format ({"worlds": [[{object_name, pose,
    # scale}, ...], ...]}). Enabled by ``record_initial_states`` in the config.
    eval_results = []
    episodes_done = 0
    with (
        contextlib.suppress(KeyboardInterrupt),  # and torch.inference_mode(),
    ):
        for batch_idx in tqdm.tqdm(range(num_batches)):
            record_video = usr_args.get("record_video", usr_args.get("env_cfg", {}).get("video", False))
            eval_video_path = Path(f"./eval_result/test_{batch_idx}/record_video.mp4") if record_video else None
            if eval_video_path is not None:
                eval_video_path.parent.mkdir(parents=True, exist_ok=True)

            usr_args['save_path'] = f"./eval_result/test_{batch_idx}"
            video_context = contextlib.nullcontext(eval_video_path)
            with video_context as v:
                obs, _reset_info = env.reset()

                policy.reset_model()
                result = policy.eval(env, obs, usr_args, v)
                eval_results.append(result)
                if getattr(policy, "_video_writer", None) is not None:
                    policy._video_writer.__exit__(None, None, None)
                    policy._video_writer = None

                take = min(num_envs, test_num - episodes_done)
                episodes_done += take
                print(f"Batch {batch_idx}: finished {take}. Total tested: {episodes_done}/{test_num}")
    final_results = policy.format_eval_results(eval_results, usr_args, episodes_done)
    print(f"Eval results: {final_results}")

    result_path = Path("./eval_result/eval_results.json")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)

    env.close()


if __name__ == "__main__":
    # example: "python robofinals/scripts/eval/policy.py --config policy/GR00T/deploy_policy_piper.yml \
    #           --overrides --env_cfg:task SizeSorting --env_cfg:layout robocasakitchen \
    #           --instruction  "Stack objects on counter from large to small" --test_num 10
    # run the main function
    usr_args = parse_args_and_config()
    main(usr_args)
