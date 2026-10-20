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

import json
import logging
import os
import shutil
import time
from copy import deepcopy

import torch

from robofinals.utils.profile_utils import RollingPerfLogger

_log = logging.getLogger(__name__)


def _scrub_broken_warp_kernel_cache_entries() -> None:
    """Remove Warp kernel cache subtrees with empty or invalid ``*.meta`` JSON.

    Incomplete compilation (crash, OOM, race) can leave a ``*.meta`` file empty; the next
    :func:`warp.launch` then fails inside ``json.load`` with JSONDecodeError. Deleting the
    affected cache directory forces a clean recompile on next use.

    Cache location matches Warp: ``WARP_CACHE_PATH`` or the OS user cache for NVIDIA/warp.
    """
    try:
        import appdirs
        import warp.config as warp_config
    except Exception:
        return

    if os.environ.get("WARP_CACHE_PATH"):
        cache_root = os.path.realpath(os.environ["WARP_CACHE_PATH"])
    else:
        cache_root = appdirs.user_cache_dir(appname="warp", appauthor="NVIDIA", version=warp_config.version)

    if not os.path.isdir(cache_root):
        return

    for entry in os.listdir(cache_root):
        mod_dir = os.path.join(cache_root, entry)
        if not os.path.isdir(mod_dir):
            continue
        scrub = False
        for root, _dirs, files in os.walk(mod_dir):
            for fn in files:
                if not fn.endswith(".meta"):
                    continue
                path = os.path.join(root, fn)
                try:
                    if os.path.getsize(path) == 0:
                        scrub = True
                        break
                    with open(path) as f:
                        json.load(f)
                except (OSError, json.JSONDecodeError, UnicodeError):
                    scrub = True
                    break
            if scrub:
                break
        if scrub:
            shutil.rmtree(mod_dir, ignore_errors=True)
            _log.warning(
                "Removed corrupted Warp kernel cache directory %s (empty or invalid *.meta); "
                "kernels will recompile on next launch.",
                mod_dir,
            )


_scrub_broken_warp_kernel_cache_entries()


def patch_warp_to_torch_accept_torch_tensor():
    from robofinals.utils.warp_compat import ensure_warp_config_compat

    ensure_warp_config_compat()
    import warp as wp

    orig_to_torch = wp.to_torch

    def to_torch(value, *args, **kwargs):
        if isinstance(value, torch.Tensor):
            return value
        return orig_to_torch(value, *args, **kwargs)

    wp.to_torch = to_torch


def _has_rtx_sensors(env) -> bool:
    """Return whether the env has RTX sensors across Isaac Lab API versions."""
    sim_has_rtx_sensors = getattr(getattr(env, "sim", None), "has_rtx_sensors", None)
    if callable(sim_has_rtx_sensors):
        return bool(sim_has_rtx_sensors())

    env_has_rtx_sensors = getattr(env, "has_rtx_sensors", False)
    if callable(env_has_rtx_sensors):
        return bool(env_has_rtx_sensors())
    return bool(env_has_rtx_sensors)


def _is_rendering(env) -> bool:
    """Return whether the env should render between physics steps across Isaac Lab API versions."""
    sim = getattr(env, "sim", None)
    sim_is_rendering = getattr(sim, "is_rendering", None)
    if sim_is_rendering is not None:
        return bool(sim_is_rendering() if callable(sim_is_rendering) else sim_is_rendering)

    sim_has_gui = getattr(sim, "has_gui", None)
    has_gui = bool(sim_has_gui() if callable(sim_has_gui) else sim_has_gui)
    return has_gui or _has_rtx_sensors(env)


def _sync_visual_after_state_reset(env) -> None:
    """Push state-reset physics buffers to fabric/RTX so Kit matches sim tensors."""

    env.backend_runtime.state.sync_after_state_reset(env)


def _backend_recording(env):

    return env.backend_runtime.recording


def _allow_backend_reset_render(env) -> bool:
    renderer = env.backend_runtime.renderer
    capabilities = getattr(renderer, "capabilities", None)
    if capabilities is not None:
        return bool(capabilities.allow_reset_render)
    return True


def _allow_backend_texture_wait(env) -> bool:
    renderer = env.backend_runtime.renderer
    capabilities = getattr(renderer, "capabilities", None)
    if capabilities is not None:
        return bool(capabilities.allow_texture_wait)
    return True


def _should_defer_record_obs(env) -> bool:
    """Return whether the selected backend defers policy observation capture."""
    return _backend_recording(env).should_defer_observation(env)


def _episode_value_for_storage(env, value):
    """Store recorder episode tensors on CPU for Newton to avoid export-time CUDA/Warp clashes."""
    return _backend_recording(env).prepare_value(value)


def _record_pre_reset_with_optional_discard(env, env_ids) -> None:
    """Run pre-reset recording, optionally replacing the rejected episode."""
    discard_recording = bool(
        getattr(env, "_robofinals_discard_recording_on_reset", False)
    )
    if discard_recording:
        env._robofinals_discard_recording_on_reset = False
    env.recorder_manager.record_pre_reset(
        env_ids,
        force_export_or_skip=False if discard_recording else None,
    )
    if discard_recording:
        env.recorder_manager.reset(env_ids)


def patch_reset():
    from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv

    def reset(
        self: ManagerBasedRLEnv, seed: int | None = None, env_ids=None, options=None
    ):
        """Resets the specified environments and returns observations.

        This function calls the :meth:`_reset_idx` function to reset the specified environments.
        However, certain operations, such as procedural terrain generation, that happened during initialization
        are not repeated.

        Args:
            seed: The seed to use for randomization. Defaults to None, in which case the seed is not set.
            env_ids: The environment ids to reset. Defaults to None, in which case all environments are reset.
            options: Additional information to specify how the environment is reset. Defaults to None.

                Note:
                    This argument is used for compatibility with Gymnasium environment definition.

        Returns:
            A tuple containing the observations and extras.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)

        # trigger recorder terms for pre-reset calls
        _record_pre_reset_with_optional_discard(self, env_ids)

        # set the seed
        if seed is not None:
            self.seed(seed)

        # reset state of scene
        self._reset_idx(env_ids)

        # update articulation kinematics
        self.scene.write_data_to_sim()
        self.sim.forward()
        # Match upstream ManagerBasedEnv.reset while keeping compatibility with older Isaac Lab APIs.
        num_rerenders = getattr(self.cfg, "num_rerenders_on_reset", 0)
        if num_rerenders <= 0 and getattr(self.cfg, "rerender_on_reset", False):
            num_rerenders = 1
        if _allow_backend_reset_render(self) and _has_rtx_sensors(self) and num_rerenders > 0:
            for _ in range(num_rerenders):
                self.sim.render()

        # trigger recorder terms for post-reset calls
        self.recorder_manager.record_post_reset(env_ids)

        # compute observations
        self.obs_buf = self.observation_manager.compute(update_history=True)

        if (
            _allow_backend_texture_wait(self)
            and self.cfg.wait_for_textures
            and _has_rtx_sensors(self)
        ):
            if hasattr(self.sim.physics_manager, "assets_loading"):
                while self.sim.physics_manager.assets_loading():
                    self.sim.render()
        arena_env = getattr(self.cfg, "isaaclab_arena_env", None)
        task = getattr(arena_env, "task", None) if arena_env is not None else None
        if task is not None and hasattr(task, "foreground_semantic_id_mapping"):
            task.foreground_semantic_id_mapping
            # self.cfg.setup_camera_and_foreground(self.scene)
            self.cfg.isaaclab_arena_env.task.record_semantic_id_mapping(self.scene)
        try:
            task_obj = getattr(getattr(self.cfg, "isaaclab_arena_env", None), "task", None)
            get_instruction = getattr(task_obj, "get_task_instruction", None)
            if callable(get_instruction):
                self.extras["task_instruction"] = get_instruction(self)
        except Exception:
            if not getattr(self, "_task_instruction_warned", False):
                import traceback
                self._task_instruction_warned = True
                print(
                    "[WARN][monkey_patch] get_task_instruction failed; task_instruction will be missing:\n"
                    + traceback.format_exc(),
                    flush=True,
                )

        return self.obs_buf, self.extras
    ManagerBasedRLEnv.reset = reset

# monkey patch the configclass to allow validate dict with key is not a string


# def patch_configclass():
#     from isaaclab.utils.configclass import configclass
#     import sys
#     configclass_module = sys.modules["isaaclab.utils.configclass"]

#     orig_validate = configclass_module._validate

#     def _validate_with_dict_key_not_string(obj, prefix=""):
#         if isinstance(obj, dict):
#             if any(not isinstance(key, str) for key in obj.keys()):
#                 obj = {str(key): value for key, value in obj.items()}
#         return orig_validate(obj, prefix=prefix)

#     configclass_module._validate = _validate_with_dict_key_not_string


def patch_configclass():
    from isaaclab.utils.configclass import configclass
    import sys

    configclass_module = sys.modules["isaaclab.utils.configclass"]
    orig_validate = configclass_module._validate

    def _validate_with_dict_key_not_string(obj, prefix=""):
        if not hasattr(_validate_with_dict_key_not_string, '_visited'):
            _validate_with_dict_key_not_string._visited = set()
            is_top_level = True
        else:
            is_top_level = False

        try:
            obj_id = id(obj)
            if obj_id in _validate_with_dict_key_not_string._visited:
                return []
            if isinstance(obj, (dict, list, tuple)) or hasattr(obj, "__dict__"):
                _validate_with_dict_key_not_string._visited.add(obj_id)

            return orig_validate(obj, prefix=prefix)
        finally:
            if is_top_level:
                delattr(_validate_with_dict_key_not_string, '_visited')

    configclass_module._validate = _validate_with_dict_key_not_string


# monkey patch the recorder manager to have ep_meta stored in the hdf5 file
def patch_recorder_manager_ep_meta():
    from .robocasa_utils import convert_fixture_to_name
    from isaaclab.managers.recorder_manager import RecorderManager

    from robofinals.utils.lerobot_isaac_export import LEROBOT_EXPORT_HOOK_ATTR

    def get_ep_meta(mgr: RecorderManager):
        ep_meta = mgr._env.cfg.isaaclab_arena_env.orchestrator.get_ep_meta()
        for obj in ep_meta["object_cfgs"]:
            obj["placement"] = convert_fixture_to_name(obj["placement"])
        return ep_meta

    orig_export_episodes = RecorderManager.export_episodes

    def export_episodes(self, env_ids=None) -> None:
        successful_before = int(self.exported_successful_episode_count)
        if env_ids is None:
            env_ids = list(range(self._env.num_envs))
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.tolist()
        if len(self.active_terms) and any(
            (env_id in self._episodes and not self._episodes[env_id].is_empty())
            for env_id in env_ids
        ):
            ep_meta = get_ep_meta(self)
            if self._dataset_file_handler is not None:
                self._dataset_file_handler.add_env_args(ep_meta)
            if self._failed_episode_dataset_file_handler is not None:
                self._failed_episode_dataset_file_handler.add_env_args(ep_meta)

            for env_id in env_ids:
                if env_id in self._episodes and not self._episodes[env_id].is_empty():
                    self._episodes[env_id].pre_export()
        orig_export_episodes(self, env_ids)
        if getattr(self._env, "_robofinals_skip_lerobot_export_hook", False):
            return
        if int(self.exported_successful_episode_count) <= successful_before:
            return
        cb = getattr(self._env, LEROBOT_EXPORT_HOOK_ATTR, None)
        if callable(cb):
            try:
                cb(self._env)
            except TypeError:
                cb()
            except Exception as e:
                _log.warning("LeRobot flush after export_episodes: %s", e)

    RecorderManager.export_episodes = export_episodes


# patch the recorder manager to have joint targets stored in the hdf5 file
def patch_recorder_manager_joint_targets():
    from isaaclab.managers.recorder_manager import RecorderTerm, RecorderManager

    def record_term_record_pre_physics_step(self: RecorderTerm):
        return None, None
    RecorderTerm.record_pre_physics_step = record_term_record_pre_physics_step

    def recorder_manager_record_pre_physics_step(self: RecorderManager):
        if len(self.active_terms) == 0:
            return

        for term in self._terms.values():
            key, value = term.record_pre_physics_step()
            self.add_to_episodes(key, value)

    RecorderManager.record_pre_physics_step = recorder_manager_record_pre_physics_step

    # TODO: add RecorderManager.record_pre_physics_step to env step

    from isaaclab.utils.datasets.episode_data import EpisodeData

    EpisodeData._next_joint_target_index = 0

    def get_joint_target(episode_data: EpisodeData, joint_target_index) -> dict | torch.Tensor | None:
        """Get the joint target of the specified index from the dataset."""
        if "joint_targets" not in episode_data._data:
            return None

        joint_targets = episode_data._data["joint_targets"]

        def get_joint_target_helper(joint_targets, joint_target_index) -> dict | torch.Tensor | None:
            if isinstance(joint_targets, dict):
                output_joint_targets = dict()
                for key, value in joint_targets.items():
                    output_joint_targets[key] = get_joint_target_helper(value, joint_target_index)
                    if output_joint_targets[key] is None:
                        return None
            elif isinstance(joint_targets, torch.Tensor):
                if joint_target_index >= len(joint_targets):
                    return None
                output_joint_targets = joint_targets[joint_target_index]
            else:
                raise ValueError(f"Invalid joint target type: {type(joint_targets)}")
            return output_joint_targets

        output_joint_targets = get_joint_target_helper(joint_targets, joint_target_index)
        return output_joint_targets

    def get_next_joint_target(self) -> dict | torch.Tensor | None:
        """Get the next joint target from the dataset."""
        joint_target = get_joint_target(self, self._next_joint_target_index)
        if joint_target is not None:
            self._next_joint_target_index += 1
        return joint_target

    EpisodeData.get_next_joint_target = get_next_joint_target

    def get_state(self, state_index) -> dict | None:
        """Get the state of the specified index from the dataset."""
        if "states" not in self._data:
            return None

        states = self._data["states"]

        def get_state_helper(states, state_index) -> dict | torch.Tensor | None:
            if isinstance(states, dict):
                output_state = dict()
                for key, value in states.items():
                    output_state[key] = get_state_helper(value, state_index)
                    if output_state[key] is None:
                        return None
            elif isinstance(states, torch.Tensor):
                if state_index >= len(states):
                    return None
                output_state = states[state_index, None]  # fix here
            elif isinstance(states, list):
                if state_index >= len(states):
                    return None
                output_state = [states[state_index]]
            else:
                raise ValueError(f"Invalid state type: {type(states)}")
            return output_state

        output_state = get_state_helper(states, state_index)
        return output_state

    EpisodeData.get_state = get_state

    def add(self, key: str, value: torch.Tensor | dict, clone: bool = True):
        """Add a key-value pair to the dataset.

        The key can be nested by using the "/" character.
        For example: "obs/joint_pos".

        Args:
            key: The key name.
            value: The corresponding value of tensor type or of dict type.
            clone: Whether to clone the tensor value before storing it in the episode data.
        """
        # check datatype
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                self.add(f"{key}/{sub_key}", sub_value, clone=clone)
            return

        stored = value.clone() if (clone and isinstance(value, torch.Tensor)) else value
        if isinstance(stored, torch.Tensor) and stored.is_cuda:
            stored = _backend_recording(getattr(self, "_env", None)).prepare_value(stored)
        sub_keys = key.split("/")
        current_dataset_pointer = self._data
        for sub_key_index in range(len(sub_keys)):
            if sub_key_index == len(sub_keys) - 1:
                # Add value to the final dict layer
                if sub_keys[sub_key_index] not in current_dataset_pointer:
                    current_dataset_pointer[sub_keys[sub_key_index]] = [stored]
                else:
                    current_dataset_pointer[sub_keys[sub_key_index]].append(stored)
                break
            # key index
            if sub_keys[sub_key_index] not in current_dataset_pointer:
                current_dataset_pointer[sub_keys[sub_key_index]] = dict()
            current_dataset_pointer = current_dataset_pointer[sub_keys[sub_key_index]]
    EpisodeData.add = add

    def pre_export(self):
        def pre_export_helper(data):
            for key, value in data.items():
                if isinstance(value, list):
                    if not value:
                        continue
                    if isinstance(value[0], torch.Tensor):
                        stacked = torch.stack(
                            [v.detach().cpu() if v.is_cuda else v for v in value]
                        )
                        data[key] = stacked
                    else:
                        data[key] = torch.stack(value)
                elif isinstance(value, dict):
                    pre_export_helper(value)
        start_time = time.time()
        pre_export_helper(self._data)
        end_time = time.time()
        print(f"pre_export time: {end_time - start_time:.2f}s")
    EpisodeData.pre_export = pre_export

    orig_add_to_episodes = RecorderManager.add_to_episodes

    def add_to_episodes_cpu_safe(self, key: str, value, *args, **kwargs):
        for episode in self._episodes.values():
            episode._env = self._env
        value = _episode_value_for_storage(self._env, value)
        return orig_add_to_episodes(self, key, value, *args, **kwargs)

    RecorderManager.add_to_episodes = add_to_episodes_cpu_safe


def patch_step():
    from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
    from isaaclab.utils.timer import Timer

    def step(self, action: torch.Tensor):
        """Execute one time-step of the environment's dynamics and reset terminated environments.

        Unlike the :class:`ManagerBasedEnv.step` class, the function performs the following operations:

        1. Process the actions.
        2. Perform physics stepping.
        3. Perform rendering if gui is enabled.
        4. Update the environment counters and compute the rewards and terminations.
        5. Reset the environments that terminated.
        6. Compute the observations.
        7. Return the observations, rewards, resets and extras.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        perf = getattr(self, "_robofinals_teleop_step_perf", None)
        if perf is None:
            perf = RollingPerfLogger("env_step")
            self._robofinals_teleop_step_perf = perf
        perf_enabled = perf.enabled
        step_started = time.perf_counter() if perf_enabled else 0.0

        action = action.to(self.device)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        # Single-env policies (PI, replay, etc.) may submit a batch of size 1. Broadcast
        # before process_action so each action term receives (num_envs, term_dim) slices.
        if action.shape[0] == 1 and self.num_envs > 1:
            action = action.repeat(self.num_envs, 1)
        # process actions
        self.action_manager.process_action(action)
        self.recorder_manager.record_pre_step()
        action_done = time.perf_counter() if perf_enabled else 0.0

        # check if we need to do rendering within the physics loop
        # note: Isaac Lab 2.3 uses sim.has_gui() / sim.has_rtx_sensors().
        is_rendering = _is_rendering(self)

        # perform physics stepping
        physics_seconds = 0.0
        render_seconds = 0.0
        scene_update_seconds = 0.0
        render_count = 0
        pending_capture_at_start = float(
            bool(getattr(self, "_robofinals_viewport_capture_pending", False))
        )
        for _ in range(self.cfg.decimation):
            physics_started = time.perf_counter() if perf_enabled else 0.0
            self._sim_step_counter += 1
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # Galbot teleop commands are held at physics rate so contact impulses
            # cannot push the free mobile base away between control updates.
            embodiment = getattr(getattr(self.cfg, "isaaclab_arena_env", None), "embodiment", None)
            apply_pre_physics_step = getattr(embodiment, "apply_pre_physics_step", None)
            if callable(apply_pre_physics_step):
                apply_pre_physics_step(self)
            # simulate
            self.sim.step(render=False)
            self.recorder_manager.record_pre_physics_step()
            if perf_enabled:
                physics_seconds += time.perf_counter() - physics_started
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                renderer = self.backend_runtime.renderer
                skip_render = renderer.should_skip_step_render(self)
                if not skip_render:
                    render_started = time.perf_counter() if perf_enabled else 0.0
                    self.sim.render()
                    render_count += 1
                    if perf_enabled:
                        render_seconds += time.perf_counter() - render_started
            # update buffers at sim dt
            scene_update_started = time.perf_counter() if perf_enabled else 0.0
            self.scene.update(dt=self.physics_dt)
            if perf_enabled:
                scene_update_seconds += time.perf_counter() - scene_update_started

        physics_done = time.perf_counter() if perf_enabled else 0.0

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_time_outs = self.termination_manager.time_outs
        self.extras["is_success"] = self.termination_manager._terminated_buf
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
        reward_done = time.perf_counter() if perf_enabled else 0.0

        if len(self.recorder_manager.active_terms) > 0:
            if not _should_defer_record_obs(self):
                self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()
        recorder_done = time.perf_counter() if perf_enabled else 0.0

        # Expose task-level debug/status vector for downstream policy logic.
        # Factory tasks implement _get_extra_status(env) for sub-phase and contact-force diagnostics.
        # get extra status before reset
        try:
            task_obj = getattr(getattr(self.cfg, "isaaclab_arena_env", None), "task", None)
            get_extra_status = getattr(task_obj, "_get_extra_status", None)
            extra_status = None
            if callable(get_extra_status):
                extra_status = get_extra_status(self)
                self.extras["extra_status"] = extra_status
            get_instruction = getattr(task_obj, "get_task_instruction", None)
            if callable(get_instruction):
                self.extras["task_instruction"] = get_instruction(self, extra_status=extra_status)
        except Exception:
            if not getattr(self, "_task_status_warned", False):
                import traceback
                self._task_status_warned = True
                print(
                    "[WARN][monkey_patch] task status export failed; extra_status/task_instruction may be missing:\n"
                    + traceback.format_exc(),
                    flush=True,
                )
        task_status_done = time.perf_counter() if perf_enabled else 0.0

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            obs_final_buf = self.observation_manager.compute(update_history=False)
            self.extras['final_obs'] = obs_final_buf

            # trigger recorder terms for pre-reset calls
            self.recorder_manager.record_pre_reset(reset_env_ids)

            self._reset_idx(reset_env_ids)
            # update articulation kinematics
            self.scene.write_data_to_sim()
            self.sim.forward()

            # if sensors are added to the scene, make sure we render to reflect changes in reset
            num_rerenders = getattr(self.cfg, "num_rerenders_on_reset", 0)
            if num_rerenders <= 0 and getattr(self.cfg, "rerender_on_reset", False):
                num_rerenders = 1
            if _has_rtx_sensors(self) and num_rerenders > 0:
                for _ in range(num_rerenders):
                    self.sim.render()

            # trigger recorder terms for post-reset calls
            self.recorder_manager.record_post_reset(reset_env_ids)

        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- step interval events
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute(update_history=True)

        if perf_enabled:
            step_done = time.perf_counter()
            perf.record(
                total=step_done - step_started,
                action_record_setup=action_done - step_started,
                physics=physics_seconds,
                render=render_seconds,
                render_count=float(render_count),
                pending_capture_count=pending_capture_at_start,
                scene_update=scene_update_seconds,
                termination_reward=reward_done - physics_done,
                recorder_post=recorder_done - reward_done,
                task_status=task_status_done - recorder_done,
                reset_events_observation=step_done - task_status_done,
            )

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_buf, self.reset_time_outs, self.extras

    ManagerBasedRLEnv.step = step

    def reset_to_check_state(
        self,
        state: dict[str, dict[str, dict[str, torch.Tensor]]],
        env_ids,
        seed: int | None = None,
        is_relative: bool = False,
    ):
        """Resets specified environments to provided states.

        This function resets the environments to the provided states. The state is a dictionary
        containing the state of the scene entities. Please refer to :meth:`InteractiveScene.get_state`
        for the format.

        The function is different from the :meth:`reset` function as it resets the environments to specific states,
        instead of using the randomization events for resetting the environments.

        Args:
            state: The state to reset the specified environments to. Please refer to
                :meth:`InteractiveScene.get_state` for the format.
            env_ids: The environment ids to reset. Defaults to None, in which case all environments are reset.
            seed: The seed to use for randomization. Defaults to None, in which case the seed is not set.
            is_relative: If set to True, the state is considered relative to the environment origins.
                Defaults to False.
        """
        # reset all envs in the scene if env_ids is None
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)

        # trigger recorder terms for pre-reset calls
        # self.recorder_manager.record_pre_reset(env_ids)

        # set the seed
        if seed is not None:
            self.seed(seed)

        previous_restoring_state = getattr(self, "_robofinals_restoring_state", False)
        self._robofinals_restoring_state = True
        try:
            self._reset_idx(env_ids)
        finally:
            self._robofinals_restoring_state = previous_restoring_state

        # set the state
        self.scene.reset_to(state, env_ids, is_relative=is_relative)

        # update articulation kinematics
        self.sim.forward()
        _sync_visual_after_state_reset(self)

        # if sensors are added to the scene, make sure we render to reflect changes in reset
        num_rerenders = getattr(self.cfg, "num_rerenders_on_reset", 0)
        if num_rerenders <= 0 and getattr(self.cfg, "rerender_on_reset", False):
            num_rerenders = 1
        if _has_rtx_sensors(self) and num_rerenders > 0:
            for _ in range(num_rerenders):
                self.sim.render()

        # trigger recorder terms for post-reset calls
        # self.recorder_manager.record_post_reset(env_ids)

        # compute observations
        self.obs_buf = self.observation_manager.compute(update_history=True)

        # return observations
        return self.obs_buf, self.extras

    ManagerBasedRLEnv.reset_to_check_state = reset_to_check_state


def patch_reset_to():
    from isaaclab.envs.manager_based_env import ManagerBasedEnv

    def reset_to(
        self,
        state: dict[str, dict[str, dict[str, torch.Tensor]]],
        env_ids,
        seed: int | None = None,
        is_relative: bool = False,
    ):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)

        _record_pre_reset_with_optional_discard(self, env_ids)

        if seed is not None:
            self.seed(seed)

        previous_restoring_state = getattr(self, "_robofinals_restoring_state", False)
        self._robofinals_restoring_state = True
        try:
            self._reset_idx(env_ids)
        finally:
            self._robofinals_restoring_state = previous_restoring_state
        self.scene.reset_to(state, env_ids, is_relative=is_relative)
        self.sim.forward()
        _sync_visual_after_state_reset(self)

        num_rerenders = getattr(self.cfg, "num_rerenders_on_reset", 0)
        if num_rerenders <= 0 and getattr(self.cfg, "rerender_on_reset", False):
            num_rerenders = 1
        if _has_rtx_sensors(self) and num_rerenders > 0:
            for _ in range(num_rerenders):
                self.sim.render()

        self.recorder_manager.record_post_reset(env_ids)
        self.obs_buf = self.observation_manager.compute(update_history=True)
        return self.obs_buf, self.extras

    ManagerBasedEnv.reset_to = reset_to


YAML_CACHE = {}


def patch_yaml_load():
    import yaml

    orig_safe_load = yaml.safe_load

    def cached_safe_load(stream):
        if isinstance(stream, (str, bytes)):
            cache_key = stream
        else:
            cache_key = stream.name
        # print(f"yaml load {cache_key}")
        if cache_key not in YAML_CACHE:
            # print(f"cache miss {cache_key}")
            YAML_CACHE[cache_key] = orig_safe_load(stream)
        return deepcopy(YAML_CACHE[cache_key])

    yaml.safe_load = cached_safe_load


def patch_reward_manager():
    from isaaclab.managers.reward_manager import RewardManager

    def compute(self, dt: float) -> torch.Tensor:
        """Computes the reward signal as a weighted sum of individual terms.

        This function calls each reward term managed by the class and adds them to compute the net
        reward signal. It also updates the episodic sums corresponding to individual reward terms.

        Args:
            dt: The time-step interval of the environment.

        Returns:
            The net reward signal of shape (num_envs,).
        """
        # reset computation
        self._reward_buf[:] = 0.0
        # iterate over all the reward terms
        for term_idx, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            # skip if weight is zero (kind of a micro-optimization)
            if term_cfg.weight == 0.0:
                self._step_reward[:, term_idx] = 0.0
                continue
            # compute term's value
            value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight
            # update total reward
            self._reward_buf += value
            # update episodic sum
            self._episode_sums[name] += value * dt

            # Update current reward for this step.
            self._step_reward[:, term_idx] = value
        # Normalization
        self._reward_buf /= 3
        return self._reward_buf

    RewardManager.compute = compute


def patch_create_teleop_device():
    """Patch Isaac Lab teleop factory for Isaac Lab 2.x / robofinals compatibility.

    Upstream removed ``DEVICE_MAP`` / ``RETARGETER_MAP``; devices are resolved from
    ``DeviceCfg.class_type`` and retargeters from ``retargeter_type`` (see
    ``isaaclab_teleop.deprecated.teleop_device_factory``). robofinals teleop scripts
    still call ``create_teleop_device(env, device_name, devices_cfg, callbacks)``;
    Isaac Lab scripts use ``create_teleop_device(device_name, devices_cfg, callbacks)``.
    """
    import inspect
    from collections.abc import Callable
    from typing import cast

    import isaaclab.devices.teleop_device_factory as teleop_device_factory
    from isaaclab.devices import DeviceBase, DeviceCfg
    from isaaclab.devices.retargeter_base import RetargeterBase

    def _build_teleop_device(
        device_name: str,
        devices_cfg: dict[str, DeviceCfg],
        callbacks: dict[str, Callable] | None,
        env,
    ) -> DeviceBase:
        if device_name not in devices_cfg:
            raise ValueError(f"Device '{device_name}' not found in teleop device configurations")

        device_cfg = devices_cfg[device_name]
        callbacks = callbacks or {}

        device_constructor = getattr(device_cfg, "class_type", None)
        if device_constructor is None:
            raise ValueError(
                f"Device configuration '{device_name}' does not declare class_type. "
                "Set cfg.class_type to the concrete DeviceBase subclass."
            )
        check_cls = device_constructor._resolve() if hasattr(device_constructor, "_resolve") else device_constructor
        if not issubclass(check_cls, DeviceBase):
            raise TypeError(
                f"class_type for '{device_name}' must be a subclass of DeviceBase; got {device_constructor}"
            )

        retargeters = []
        if hasattr(device_cfg, "retargeters") and device_cfg.retargeters is not None:
            try:
                for retargeter_cfg in device_cfg.retargeters:
                    retargeter_constructor = getattr(retargeter_cfg, "retargeter_type", None)
                    if retargeter_constructor is None:
                        raise ValueError(
                            f"Retargeter configuration {type(retargeter_cfg).__name__} does not declare retargeter_type. "
                            "Set cfg.retargeter_type to the concrete RetargeterBase subclass."
                        )
                    if not issubclass(retargeter_constructor, RetargeterBase):
                        raise TypeError(
                            f"retargeter_type for {type(retargeter_cfg).__name__} must be a subclass of RetargeterBase; got"
                            f" {retargeter_constructor}"
                        )
                    retargeters.append(retargeter_constructor(retargeter_cfg))

            except NameError as e:
                raise ValueError(f"Failed to create retargeters: {e}") from e

        constructor_params = inspect.signature(device_constructor).parameters
        params: dict = {"cfg": device_cfg}
        if "retargeters" in constructor_params:
            params["retargeters"] = retargeters
        if "env" in constructor_params:
            params["env"] = env

        device = cast(DeviceBase, device_constructor(**params))

        for key, callback in callbacks.items():
            device.add_callback(key, callback)

        _log.info("Created teleoperation device: %s", device_name)
        return device

    def create_teleop_device(*args, **kwargs) -> DeviceBase:
        # Isaac Lab: (device_name, devices_cfg[, callbacks])
        if args and isinstance(args[0], str):
            device_name, devices_cfg = args[0], args[1]
            callbacks = args[2] if len(args) > 2 else kwargs.get("callbacks")
            return _build_teleop_device(device_name, devices_cfg, callbacks, env=None)
        # robofinals legacy: (env, device_name, devices_cfg[, callbacks])
        if len(args) < 3:
            raise TypeError(
                "create_teleop_device expected (device_name, devices_cfg[, callbacks]) "
                "or (env, device_name, devices_cfg[, callbacks])"
            )
        env, device_name, devices_cfg = args[0], args[1], args[2]
        callbacks = args[3] if len(args) > 3 else kwargs.get("callbacks")
        return _build_teleop_device(device_name, devices_cfg, callbacks, env=env)

    teleop_device_factory.create_teleop_device = create_teleop_device


def patch_isaaclab_tasks_mdp():
    """Patch isaaclab_tasks.manager_based.manipulation.pick_place.mdp to include functions from robofinals.

    This ensures that functions like get_eef_pos, get_eef_quat, and get_robot_joint_state
    are available in the isaaclab_tasks.mdp module even though they were moved to robofinals.

    Uses a module loader wrapper to patch the module immediately after it's loaded.
    """
    import sys
    import importlib.util
    import importlib.machinery

    module_name = "isaaclab_tasks.manager_based.manipulation.pick_place.mdp"

    def _patch_module(module):
        """Patch the module with our functions."""
        try:
            from robofinals.core.mdp.observations import (
                get_eef_pos,
                get_eef_quat,
                get_robot_joint_state,
            )

            # Inject our functions into the isaaclab_tasks.mdp module
            # These functions were removed from isaaclab_tasks but are still needed by gr1t2.py
            module.get_eef_pos = get_eef_pos
            module.get_eef_quat = get_eef_quat
            module.get_robot_joint_state = get_robot_joint_state
        except (ImportError, AttributeError):
            # Silently fail if we can't patch
            pass

    # Patch immediately if module is already loaded
    if module_name in sys.modules:
        _patch_module(sys.modules[module_name])

    # Set up a meta path finder to intercept and patch the module
    class MdpPatcher:
        """Meta path finder that patches the mdp module after import."""

        def find_spec(self, name, path, target=None):
            if name != module_name:
                return None

            # Find the spec using other finders
            for finder in sys.meta_path:
                if finder is self:
                    continue
                spec = finder.find_spec(name, path)
                if spec is not None and spec.loader is not None:
                    # Wrap the loader to patch after module execution
                    original_exec_module = spec.loader.exec_module

                    def patched_exec_module(module):
                        original_exec_module(module)
                        # Patch after module is loaded
                        _patch_module(module)

                    spec.loader.exec_module = patched_exec_module
                    return spec
            return None

    # Only register the finder once
    if not hasattr(sys, '_robofinals_mdp_patcher_registered'):
        patcher = MdpPatcher()
        sys.meta_path.insert(0, patcher)
        sys._robofinals_mdp_patcher_registered = True



def patch_camera_env_mask_has_any():
    """Skip Warp→NumPy D2H when checking whether any env is selected in a camera mask."""
    from isaaclab.sensors.camera.camera import Camera

    if getattr(Camera, "_robofinals_env_mask_patched", False):
        return

    @staticmethod
    def _env_mask_has_any(env_mask):
        return int(env_mask.size) > 0

    Camera._env_mask_has_any = _env_mask_has_any
    Camera._robofinals_env_mask_patched = True


def apply_all_patches() -> None:
    """Apply the compatibility set explicitly after Kit/AppLauncher startup."""

    patch_warp_to_torch_accept_torch_tensor()
    patch_camera_env_mask_has_any()
    patch_reset()
    patch_reset_to()
    patch_configclass()
    patch_recorder_manager_ep_meta()
    patch_recorder_manager_joint_targets()
    patch_step()
    patch_yaml_load()
    patch_reward_manager()
    patch_create_teleop_device()
    patch_isaaclab_tasks_mdp()
