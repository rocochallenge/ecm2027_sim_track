"""Evaluator-side supervisor for one locally launched policy process group."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def all_environments_waiting_after_wall_timeout(run_dir, *, wall_timeout_seconds=300*60):
    """Check ended timeouts or elapsed live episodes, even if simulator step stalls."""
    run_dir=Path(run_dir)
    now=time.monotonic()
    try:
        count=int(json.loads((run_dir/"run.json").read_text())["num_envs"])
        if count<1: return False
        for i in range(count):
            episodes=sorted((run_dir/f"env_{i:04d}").glob("episode_*"))
            if not episodes: return False
            latest=episodes[-1]
            manifest=json.loads((latest/"manifest.json").read_text())
            if manifest.get("status")=="recording":
                started=manifest.get("started_monotonic")
                if started is None or now-float(started)<wall_timeout_seconds: return False
            elif manifest.get("status") in ("complete","incomplete"):
                score=json.loads((latest/"score.json").read_text())
                if score.get("end_reason")!="wall_clock_timeout": return False
            else:
                return False
        return True
    except (OSError,ValueError,KeyError,TypeError,json.JSONDecodeError):
        return False


def _terminate_owned_group(process, stop_seconds):
    if process.poll() is not None: return
    try: os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError: return
    try: process.wait(timeout=stop_seconds)
    except subprocess.TimeoutExpired:
        try: os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError: pass
        process.wait(timeout=stop_seconds)


def run_policy(command, run_dir, *, wall_timeout_seconds=300*60, grace_seconds=10.0,
               poll_seconds=1.0, stop_seconds=10.0):
    if not command: raise ValueError("policy command is required")
    if wall_timeout_seconds<=0 or grace_seconds<0 or poll_seconds<=0 or stop_seconds<=0:
        raise ValueError("invalid supervisor timing")
    if not (Path(run_dir)/"run.json").is_file(): raise FileNotFoundError("run.json not found in run directory")
    process=subprocess.Popen(command,start_new_session=True)
    pending_since=None
    try:
        while process.poll() is None:
            if all_environments_waiting_after_wall_timeout(run_dir, wall_timeout_seconds=wall_timeout_seconds):
                if pending_since is None: pending_since=time.monotonic()
                if time.monotonic()-pending_since>=grace_seconds:
                    print("[policy_supervisor] all environments timed out; stopping policy",flush=True)
                    _terminate_owned_group(process,stop_seconds)
                    return 124
            else:
                pending_since=None
            time.sleep(poll_seconds)
        return process.returncode
    finally:
        _terminate_owned_group(process,stop_seconds)


def main():
    parser=argparse.ArgumentParser(description="Run a local policy and stop it if all environments time out.")
    parser.add_argument("--run-dir",required=True,help="Evaluator's episode run directory containing run.json")
    parser.add_argument("--grace-seconds",type=float,default=10.0)
    parser.add_argument("--wall-timeout-minutes",type=float,default=300.0)
    parser.add_argument("policy_command",nargs=argparse.REMAINDER)
    args=parser.parse_args()
    command=args.policy_command
    if command and command[0]=="--": command=command[1:]
    return run_policy(command,args.run_dir,wall_timeout_seconds=args.wall_timeout_minutes*60,
                      grace_seconds=args.grace_seconds)

if __name__=="__main__": sys.exit(main())
