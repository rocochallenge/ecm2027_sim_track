"""PhysX-specific composition for teleop USD animation recording."""

from __future__ import annotations

import os

from robofinals.core.backends.physx.rigid_animation import PhysXRigidAnimationRecorder

from .ovd_recorder import OvdRecorderController


class PhysxRecorderController(OvdRecorderController):
    def __init__(self, args_cli, logger_getter):
        super().__init__(args_cli, logger_getter)
        self.rigid = PhysXRigidAnimationRecorder(self.logger)
        self._rigid_sample_error_reported = False

    def prepare(self, env):
        self._rigid_sample_error_reported = False
        super().prepare(env)
        if self.enabled:
            self._prepare_rigid(env)

    def _prepare_rigid(self, env):
        try:
            self.rigid.prepare(env)
        except Exception as exc:
            self.rigid.reset()
            self.logger.warning(
                "[AnimRecording][PhysX] Failed to prepare scene transform recording: "
                f"{type(exc).__name__}: {exc}"
            )

    def request_start(self, env):
        if not self.enabled or self.clip_started:
            return
        # A reset can rebuild IsaacLab asset data objects between setup and B.
        self._prepare_rigid(env)
        super().request_start(env)
        self.rigid.mark_start(env)
        self._sample_rigid(env)

    def sample(self, env):
        if not self.enabled or not self.clip_started:
            return
        self._sample_rigid(env)
        super().sample(env)

    def _sample_rigid(self, env):
        try:
            self.rigid.sample(env)
        except Exception as exc:
            if self._rigid_sample_error_reported:
                return
            self._rigid_sample_error_reported = True
            self.logger.warning(
                "[AnimRecording][PhysX] Failed to sample scene transforms: "
                f"{type(exc).__name__}: {exc}"
            )

    def clear_pending_if_not_started(self):
        super().clear_pending_if_not_started()
        if not self.clip_started:
            self.rigid.reset()

    def finish(self, env):
        if self.baked or not self.enabled:
            return
        if self.clip_started:
            self._sample_rigid(env)
            output_dir = self._resolve_ovd_output_dir(env)
            if output_dir is None:
                self.logger.warning(
                    "[AnimRecording][PhysX] OVD output directory is unavailable; "
                    "scene transform overlay was not written."
                )
            else:
                try:
                    self.rigid.finish(output_dir)
                except Exception as exc:
                    self.logger.warning(
                        "[AnimRecording][PhysX] Failed to write scene transform overlay: "
                        f"{type(exc).__name__}: {exc}"
                    )
        super().finish(env)

    @staticmethod
    def _resolve_ovd_output_dir(env) -> str | None:
        sim = getattr(env, "sim", None)
        output_dir = getattr(sim, "_anim_recording_output_dir", None)
        if output_dir:
            return os.path.abspath(str(output_dir))
        try:
            from isaaclab_physx.physics.physx_manager import PhysxManager

            recorder = getattr(PhysxManager, "_anim_recorder", None)
            output_dir = getattr(recorder, "_output_dir", None)
            return os.path.abspath(str(output_dir)) if output_dir else None
        except ImportError:
            return None
