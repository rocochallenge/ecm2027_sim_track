import os
import signal
import time

from .deformable_recorder import DeformableAnimationRecorder
from .usd_playback import UsdPlaybackBuilder


class OvdRecorderController:
    def __init__(self, args_cli, logger_getter):
        self.args_cli = args_cli
        self._logger_getter = logger_getter
        self.deformable = DeformableAnimationRecorder(args_cli, logger_getter)
        self.playback_builder = UsdPlaybackBuilder(logger_getter)
        self.reset_state()

    @property
    def logger(self):
        return self._logger_getter()

    @property
    def enabled(self):
        return bool(getattr(self.args_cli, "anim_recording_enabled", False))

    @property
    def clip_started(self):
        return self.clip_start_time is not None

    def reset_state(self):
        self.clip_start_time = None
        self.clip_start_step = None
        self.clip_start_env_time = None
        self.baked = False
        self.pending_start = False
        self.warmup_steps_remaining = 0

    def prepare(self, env):
        self.reset_state()
        self.deformable.reset()

        if not self.enabled:
            return

        sim = getattr(env, "sim", None)
        if sim is None or not hasattr(sim, "_anim_recording_stop_time"):
            self.logger.warning("[AnimRecording] OVD recorder was requested, but simulation does not expose OVD hooks.")
            return

        sim._anim_recording_stop_time = 999999999.0
        self.deformable.prepare(env)
        self.logger.info(
            "[AnimRecording] OVD recording armed. Keep Fabric enabled; the clip will bake automatically when teleop exits."
        )

    def request_start(self, env):
        if not self.enabled:
            return
        if self.clip_started:
            return

        self.pending_start = False
        self.warmup_steps_remaining = 0
        self.mark_start(env)
        self.deformable.mark_start(
            env,
            clip_start_time=self.clip_start_time,
            clip_start_step=self.clip_start_step,
            clip_start_env_time=self.clip_start_env_time,
        )

    def maybe_start_after_warmup(self, env):
        return

    def mark_start(self, env):
        if self.clip_started:
            return
        if not self.enabled:
            return

        sim = getattr(env, "sim", None)
        if sim is None:
            return

        self.clip_start_time = 0.0
        self.clip_start_step, self.clip_start_env_time = self._get_env_step_time(env)

        if hasattr(sim, "_anim_recording_stop_time"):
            sim._anim_recording_stop_time = 999999999.0

        self.logger.info(
            "[AnimRecording] Teleop clip start marked at "
            f"{self.clip_start_time:.2f}s "
            f"(env_step={self.clip_start_step}, env_time={self.clip_start_env_time:.4f}s)."
        )

    def sample(self, env):
        self.deformable.sample(env)

    def reset_deformable(self):
        self.deformable.reset()

    def clear_pending_if_not_started(self):
        if self.clip_started:
            return
        self.clip_start_step = None
        self.clip_start_env_time = None
        self.pending_start = False
        self.warmup_steps_remaining = 0
        self.deformable.reset()

    def finish(self, env):
        if self.baked or not self.enabled:
            return
        if self.clip_start_time is None:
            self.logger.info("[AnimRecording] Teleop clip never started; skipping OVD bake.")
            self.pending_start = False
            self.warmup_steps_remaining = 0
            self.baked = True
            return

        sim = getattr(env, "sim", None)
        if sim is None or not hasattr(sim, "_finish_anim_recording"):
            return

        clip_stop = self._get_elapsed_time(env)
        if clip_stop is None:
            self.logger.info("[AnimRecording] No simulation steps were captured; skipping OVD bake.")
            self.baked = True
            return

        clip_start = self.clip_start_time if self.clip_start_time is not None else 0.0
        clip_stop = max(clip_start + 1e-3, clip_stop)
        sim._anim_recording_start_time = clip_start
        sim._anim_recording_stop_time = clip_stop
        sim._anim_recording_enabled = False

        self.logger.info(f"[AnimRecording] Finishing OVD baking for teleop clip [{clip_start:.2f}, {clip_stop:.2f}]s...")
        previous_sigint_handler = signal.getsignal(signal.SIGINT)
        try:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            sim._finish_anim_recording()
        finally:
            signal.signal(signal.SIGINT, previous_sigint_handler)

        from robofinals.core.backends.physx.animation_recording import repair_scene_references

        repaired_output_dir = repair_scene_references(sim._anim_recording_output_dir, env)
        combined_output_path = self.deformable.finish(repaired_output_dir)
        if combined_output_path is None:
            combined_output_path = self.playback_builder.build(repaired_output_dir)

        self.logger.info(f"[AnimRecording] OVD output: {repaired_output_dir}")
        if combined_output_path is not None:
            self.logger.info(f"[AnimRecording] Playback entry: {combined_output_path}")
        self.baked = True
        self._package_delivery(repaired_output_dir)
        if env is not None:
            env._anim_recording_baked_path = os.path.join(
                repaired_output_dir, "baked_animation_recording.usda"
            )
            delivery_dir = getattr(self.args_cli, "anim_recording_output_dir", None)
            if delivery_dir:
                env._anim_recording_baked_delivery_entry = os.path.join(
                    os.path.abspath(os.path.expanduser(delivery_dir)),
                    "teleop_baked_with_deformable.usda",
                )
            else:
                env._anim_recording_baked_delivery_entry = os.path.join(
                    repaired_output_dir.rstrip(os.sep) + "_delivery",
                    "teleop_baked_with_deformable.usda",
                )

    def repair_references(self, anim_output_dir, env):
        from robofinals.core.backends.physx.animation_recording import repair_scene_references

        return repair_scene_references(anim_output_dir, env)

    def _package_delivery(self, repaired_output_dir):
        from .package_bake_delivery import package_bake_delivery

        entry_usd = os.path.join(repaired_output_dir, "teleop_baked_with_deformable.usda")
        if os.path.exists(entry_usd):
            user_delivery_root = getattr(self.args_cli, "anim_recording_output_dir", None)
            if user_delivery_root:
                delivery_dir = os.path.abspath(os.path.expanduser(user_delivery_root))
            else:
                delivery_dir = repaired_output_dir.rstrip(os.sep) + "_delivery"
            self.logger.info(f"[AnimRecording] Packaging delivery folder -> {delivery_dir}")
            package_bake_delivery(repaired_output_dir, delivery_dir, logger=self.logger)
            delivery_entry = os.path.join(delivery_dir, "teleop_baked_with_deformable.usda")
            self.logger.info(f"[AnimRecording] Delivery entry: {delivery_entry}")
            self.logger.info("[anim_recording] ===== Animation recording complete =====")
            self.logger.info(f"[anim_recording] Raw bake folder : {repaired_output_dir}")
            self.logger.info(f"[anim_recording] Delivery folder   : {delivery_dir}")
            self.logger.info(f"[anim_recording] Open this file    : {delivery_entry}")
            self.logger.info("[anim_recording] ========================================")
        else:
            self.logger.info("[AnimRecording] Skipping delivery packaging: no teleop_baked_with_deformable.usda found.")

    def _get_elapsed_time(self, env):
        step_dt = float(getattr(env, "step_dt", 0.0) or 0.0)
        steps = int(getattr(env, "common_step_counter", 0) or 0)
        if step_dt > 0.0 and steps > 0:
            return max(1e-3, steps * step_dt)

        sim = getattr(env, "sim", None)
        started_timestamp = getattr(sim, "_anim_recording_started_timestamp", None) if sim is not None else None
        if started_timestamp is None:
            return None
        return max(0.0, time.time() - started_timestamp)

    def _get_env_step_time(self, env):
        step_count = int(getattr(env, "common_step_counter", 0) or 0)
        step_dt = float(getattr(env, "step_dt", 0.0) or 0.0)
        return step_count, step_count * step_dt
