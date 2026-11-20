import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
import cv2

class CrashVideoTests(unittest.TestCase):
    def test_encoder_finalizes_after_simulator_sigkill(self):
        code = """
import sys,time
import numpy as np
from robofinals.applications.eval.episode_records import CrashSafeVideoWriter
w=CrashSafeVideoWriter(sys.argv[1],32,32,50)
for i in range(20): w.write(np.full((32,32,3),i,dtype=np.uint8))
print('READY',flush=True)
time.sleep(60)
"""
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"crash.mp4"
            process=subprocess.Popen([sys.executable,"-u","-c",code,str(path)],stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE,text=True)
            try:
                self.assertEqual(process.stdout.readline().strip(),"READY")
                process.kill(); process.wait(timeout=5)
                deadline=time.monotonic()+15
                while time.monotonic()<deadline:
                    cap=cv2.VideoCapture(str(path))
                    frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); ok,_=cap.read(); cap.release()
                    if ok and frames>=20: break
                    time.sleep(0.2)
                self.assertTrue(ok)
                self.assertEqual(frames,20)
            finally:
                if process.poll() is None: process.kill(); process.wait(timeout=5)
                process.stdout.close(); process.stderr.close()

if __name__ == "__main__": unittest.main()
