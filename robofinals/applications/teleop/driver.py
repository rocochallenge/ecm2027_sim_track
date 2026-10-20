"""Teleoperation driver implementing the generic SessionDriver port."""

from __future__ import annotations

import time
from typing import Any, Protocol

from robofinals.core.session import IdleCommand, SessionContext, SessionTransition, StepCommand, StopCommand

from .commands import CommandMailbox


class ActionSource(Protocol):
    def reset(self) -> None: ...

    def read(self) -> Any: ...

    def close(self) -> None: ...


class LegacyDeviceActionSource:
    """Adapt existing IsaacLab devices exposing ``advance()`` to ActionSource."""

    def __init__(
        self,
        device: Any,
        simulation_app: Any = None,
        active_getter: Any = None,
        hold_action_getter: Any = None,
    ) -> None:
        self.device = device
        self.simulation_app = simulation_app
        self.active_getter = active_getter
        self.hold_action_getter = hold_action_getter

    def reset(self) -> None:
        reset = getattr(self.device, "reset", None)
        if callable(reset):
            # Bootstrap may intentionally arm a device before the runner opens
            # (for example Newton's explicit auto-start diagnostic). Preserve
            # that latch across the runner-owned reset; a normal stopped VR
            # device remains stopped.
            # Some devices (e.g., GalbotKeyboard) set preserve_started_on_reset=False
            # to require explicit re-activation (B key press) after reset.
            was_started = bool(getattr(self.device, "started", False))
            reset()
            preserve = getattr(self.device, "preserve_started_on_reset", True)
            if was_started and preserve and hasattr(self.device, "started"):
                self.device.started = True

    def is_running(self) -> bool:
        checker = getattr(self.simulation_app, "is_running", None)
        is_exiting = getattr(self.simulation_app, "is_exiting", None)
        try:
            if callable(is_exiting) and is_exiting():
                return False
            return bool(checker()) if callable(checker) else True
        except ReferenceError:
            # Kit may invalidate its application proxy while a device callback
            # is still unwinding during window shutdown.
            return False

    def read(self) -> Any:
        if not self.is_running():
            return None
        active = not callable(self.active_getter) or bool(self.active_getter())
        poll_when_inactive = bool(getattr(self.device, "poll_when_inactive", False))
        if not active and not poll_when_inactive:
            time.sleep(0.002)
            return None
        if self.device is None:
            if callable(self.hold_action_getter):
                hold_action = self.hold_action_getter()
                return hold_action.clone() if callable(getattr(hold_action, "clone", None)) else hold_action
            return None
        advance = getattr(self.device, "advance", None)
        if not callable(advance):
            raise TypeError(f"teleop device {type(self.device).__name__} does not implement advance()")
        sample = advance()
        # Poll VR button edges while inactive, but do not step until the queued
        # start/reset command has been handled by the runner.
        if not active and poll_when_inactive:
            time.sleep(0.001)
            return None
        if sample is None or (isinstance(sample, bool) and not sample):
            # Keep the Kit listener responsive while VR is waiting for B/pose input.
            time.sleep(0.001)
            if isinstance(sample, bool) and not sample and callable(self.hold_action_getter):
                # An explicitly auto-started session may keep the robot at its
                # current action until a physical device produces input.
                hold_action = self.hold_action_getter()
                if hold_action is not None:
                    return hold_action.clone() if callable(getattr(hold_action, "clone", None)) else hold_action
        return sample

    def close(self) -> None:
        close = getattr(self.device, "close", None)
        if callable(close):
            close()


class TeleopDriver:
    """Convert device samples and callback commands into generic session commands."""

    def __init__(self, source: ActionSource, mailbox: CommandMailbox | None = None) -> None:
        self.source = source
        self.mailbox = mailbox if mailbox is not None else CommandMailbox()

    def open(self, _context: SessionContext) -> None:
        print("[teleop] device driver open", flush=True)
        self.source.reset()
        print("[teleop] device driver open complete", flush=True)

    def next_command(self, _context: SessionContext):
        is_running = getattr(self.source, "is_running", None)
        if callable(is_running) and not is_running():
            print("[teleop] SessionRunner stop: simulation_app.is_running() returned False", flush=True)
            return StopCommand("simulation_closed")
        queued = self.mailbox.drain_one()
        if queued is not None:
            return queued
        try:
            action = self.source.read()
        except BaseException as exc:
            print(
                f"[teleop] device source.read failed: {type(exc).__name__}: {exc}",
                flush=True,
            )
            raise
        # Closing a Kit window can happen inside a blocking device.advance().
        # Do not dispatch the sample after PhysX has released its simulation
        # views, even if advance() returned a valid-looking action.
        if callable(is_running) and not is_running():
            print(
                "[teleop] SessionRunner stop: simulation_app closed during device read",
                flush=True,
            )
            return StopCommand("simulation_closed")
        if not hasattr(self, "_debug_read_count"):
            self._debug_read_count = 0
        if self._debug_read_count < 5:
            print(f"[RecorderDebug] driver read active={self.source.active_getter() if callable(getattr(self.source, "active_getter", None)) else None} action_type={type(action).__name__} action_none={action is None}", flush=True)
            self._debug_read_count += 1
        # Some VR backends return bool status/event values while callbacks
        # publish the real command through the mailbox. Never send a bool to env.step.
        if action is None or isinstance(action, bool):
            return IdleCommand("device_not_ready")
        return StepCommand(action)

    def on_transition(self, _context: SessionContext, _transition: SessionTransition) -> None:
        return None

    def close(self, _context: SessionContext) -> None:
        self.source.close()
