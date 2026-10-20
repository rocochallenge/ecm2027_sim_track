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

"""
Base Policy Class
Define common interfaces and functionality for all policies
"""

import numpy as np
import torch
import mediapy as media
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, Any, Union, Tuple, Sequence


class BasePolicy(ABC):
    """
    Base Policy Class
    All policies should inherit from this class
    """

    def __init__(self, usr_args: Dict[str, Any]):
        """
        Initialize policy

        Args:
            usr_args: User parameter configuration
        """
        self.usr_args = usr_args
        self.model = None
        self.instruction = None
        self._is_initialized = False
        self._video_writer = None
        self.initialize()

    def initialize(self) -> None:
        """Initialize policy model"""
        if not self._is_initialized:
            self.get_model(self.usr_args)
            self.get_instruction()
            self._is_initialized = True

    # ==================== Common Utility Methods ====================

    def get_instruction(self):
        """Get task instruction"""
        self.instruction = self.usr_args.get('instruction', '')

    def add_video_frame(self, video_writer: Any, obs: Dict[str, Any],
                        camera_key: str) -> None:
        """Add video frame.

        If ``video_writer`` is a path, initialize the writer lazily from the first
        rendered frame so recorded resolution follows the actual camera output.
        """
        if video_writer is None:
            return
        camera_images = [
            obs['policy'][[key for key in obs['policy'].keys() if key.startswith(cam)][0]].cpu().numpy()[0]
            for cam in camera_key if any(key.startswith(cam) for key in obs['policy'].keys())
        ]
        if not camera_images:
            return
        combined_image = np.concatenate(camera_images, axis=1)
        if isinstance(video_writer, (str, Path)):
            if not hasattr(self, "_video_writer") or self._video_writer is None:
                self._video_writer = media.VideoWriter(path=Path(video_writer), shape=combined_image.shape[:2], fps=30)
                self._video_writer.__enter__()
            video_writer = self._video_writer
        video_writer.add_image(combined_image)

    def step_environment(self, task_env: Any, action: Union[np.ndarray, torch.Tensor],
                         usr_args: Dict[str, Any]) -> Tuple[Dict[str, Any], bool, Dict[str, Any]]:
        """Execute environment step"""

        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float().cuda()

        if 'joint_mapping' in usr_args:
            # action = action.squeeze(0)
            action = action[..., usr_args['joint_mapping']]

        obs, _, terminated, _, extras = task_env.step(action)
        # Return the raw per-env termination signal. Callers that only need a
        # scalar can reduce it (e.g. GR00T uses _to_terminated_flag); batched
        # callers (multi-env eval) need the per-env values.
        return obs, terminated, extras

    def encode_obs(self, observation: Dict[str, Any], transpose: bool = True, keep_dim_env: bool = False,
                   env_idx: int = 0) -> Dict[str, Any]:
        """
        Preprocess observation data
        Process tensor observations and reshape dimensions

        Args:
            observation: Raw observation data

        Returns:
            Processed observation data
        """
        merged_dict = observation['policy'].copy()
        merged_dict.update(observation['embodiment_general_obs'])
        observation = merged_dict
        for name, obs in observation.items():
            # Process tensor observation data
            if torch.is_tensor(obs):
                obs_np = obs.cpu().numpy()
            else:
                obs_np = obs
            if not keep_dim_env:
                obs_np = obs_np[env_idx]

            # Reshape observation data dimensions
            if len(obs_np.shape) == 4:  # (1, H, W, C)
                observation[name] = np.transpose(obs_np, (2, 0, 1)) if transpose else obs_np
            elif len(obs_np.shape) == 3:
                if obs_np.shape[-1] == 3:
                    observation[name] = np.transpose(obs_np, (2, 0, 1)) if transpose else obs_np
                else:
                    observation[name] = obs_np
            elif len(obs_np.shape) == 2:
                observation[name] = obs_np
            else:
                observation[name] = obs_np
        return observation

    def format_eval_results(
        self,
        results: Sequence[Any],
        usr_args: Dict[str, Any],
        episodes_done: int,
    ) -> Dict[str, Any]:
        """Convert raw per-batch eval results into the JSON summary saved by eval_policy."""
        success_count = 0
        counted = 0
        per_episode = []
        for result in results:
            if torch.is_tensor(result):
                result = result.detach().cpu().numpy()
            success_mask = np.atleast_1d(np.asarray(result).astype(bool))
            remaining = max(episodes_done - counted, 0)
            take = min(success_mask.shape[0], remaining)
            for i in range(take):
                per_episode.append(
                    {"episode": counted + i, "success": bool(success_mask[i])}
                )
            success_count += int(success_mask[:take].sum())
            counted += take
            if counted >= episodes_done:
                break

        success_rate = success_count / episodes_done if episodes_done else 0.0
        return {
            "test_count": episodes_done,
            "success_count": success_count,
            "success_rate": success_rate,
            "episodes": per_episode,
        }

    # ==================== Abstract Methods - Must be implemented by subclasses ====================

    @abstractmethod
    def get_model(self, usr_args: Dict[str, Any]) -> Any:
        """
        Get model instance

        Args:
            usr_args: User parameter configuration

        Returns:
            Model instance
        """
        pass

    @abstractmethod
    def get_action(self) -> Any:
        """
        Get action from model
        """
        pass

    @abstractmethod
    def eval(self, task_env: Any, observation: Dict[str, Any],
             usr_args: Dict[str, Any], video_writer: Any) -> bool:
        """
        Evaluate policy

        Args:
            task_env: Task environment
            observation: Observation data
            usr_args: User parameters
            video_writer: Video writer

        Returns:
            Whether the task was completed successfully
        """
        pass

    @abstractmethod
    def reset_model(self) -> None:
        """
        Reset model state

        Args:
            model: Model instance
        """
        pass

    # ==================== Unified Interface Methods ====================
    def get_policy_name(self) -> str:
        """Get policy name"""
        return self.__class__.__name__.replace('Policy', '')
