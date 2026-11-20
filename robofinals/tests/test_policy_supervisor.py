import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from robofinals.applications.eval.policy_supervisor import all_environments_waiting_after_wall_timeout, run_policy

class SupervisorTests(unittest.TestCase):
    def test_cleanup_only_when_all_environments_timed_out(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/"run.json").write_text(json.dumps({"num_envs":2}))
            def write_episode(i,status,reason):
                p=root/f"env_{i:04d}"/"episode_000000";p.mkdir(parents=True,exist_ok=True)
                (p/"score.json").write_text(json.dumps({"end_reason":reason}))
                (p/"manifest.json").write_text(json.dumps({"status":status}))
            write_episode(0,"complete","wall_clock_timeout")
            write_episode(1,"recording","")
            self.assertFalse(all_environments_waiting_after_wall_timeout(root))
            def trigger():
                time.sleep(.15);write_episode(1,"complete","wall_clock_timeout")
            thread=threading.Thread(target=trigger);thread.start()
            start=time.monotonic()
            result=run_policy([sys.executable,"-c","import time;time.sleep(60)"],root,
                              grace_seconds=.1,poll_seconds=.02,stop_seconds=2)
            thread.join()
            self.assertEqual(result,124)
            self.assertGreaterEqual(time.monotonic()-start,.2)
            self.assertLess(time.monotonic()-start,5)

    def test_elapsed_live_episodes_stop_a_stuck_policy(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/"run.json").write_text(json.dumps({"num_envs":2}))
            for i in range(2):
                p=root/f"env_{i:04d}"/"episode_000000";p.mkdir(parents=True)
                (p/"manifest.json").write_text(json.dumps({"status":"recording",
                    "started_monotonic":time.monotonic()-10}))
            result=run_policy([sys.executable,"-c","import time;time.sleep(60)"],root,
                wall_timeout_seconds=1,grace_seconds=.05,poll_seconds=.02,stop_seconds=2)
            self.assertEqual(result,124)

    def test_policy_normal_exit_is_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/"run.json").write_text(json.dumps({"num_envs":1}))
            self.assertEqual(run_policy([sys.executable,"-c","pass"],d,
                                        grace_seconds=.1,poll_seconds=.02,stop_seconds=2),0)

if __name__=="__main__": unittest.main()
