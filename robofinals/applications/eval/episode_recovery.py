"""Finalize crash manifests after the evaluator's pipe closes."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

CAMERAS=("first_person_camera_rgb","left_hand_camera_rgb","right_hand_camera_rgb")

def _encoder_running(pid_file):
    try:
        pid=int(pid_file.read_text(encoding="utf-8"))
        stat=Path(f"/proc/{pid}/stat").read_text()
        return stat.rsplit(")",1)[1].strip().split()[0] != "Z"
    except (OSError, ValueError, IndexError):
        return False


def recover(root):
    for path in sorted(Path(root).rglob("manifest.json")):
        try:
            data=json.loads(path.read_text(encoding="utf-8"))
            if data.get("status")!="recording": continue
            episode=path.parent
            deadline=time.monotonic()+60
            while time.monotonic()<deadline:
                if not any(_encoder_running(episode/(name+".encoder.pid")) for name in CAMERAS):
                    break
                time.sleep(.25)
            videos={}; errors=[]
            for name in CAMERAS:
                file=episode/(name+".mp4")
                probe=subprocess.run(["ffprobe","-v","error","-select_streams","v:0","-show_entries","stream=codec_name,width,height","-of","json",str(file)],capture_output=True,text=True)
                valid=(not _encoder_running(episode/(name+".encoder.pid")) and probe.returncode==0
                       and bool(json.loads(probe.stdout or "{}").get("streams")))
                videos[name]={"file":file.name,"readable":valid}
                if not valid: errors.append(name)
            now=datetime.now(timezone.utc).isoformat()
            events=episode/"events.jsonl"
            with events.open("a",encoding="utf-8") as f:
                f.write(json.dumps({"type":"process_interrupted","utc":now})+"\n"); f.flush(); os.fsync(f.fileno())
            data.update({"status":"interrupted","ended_utc":now,"recovery_reason":"evaluator_process_exit",
                         "events":"events.jsonl","score":None,"videos":videos,"video_errors":errors})
            temp=path.with_suffix(".json.tmp")
            with temp.open("w",encoding="utf-8") as f:
                json.dump(data,f,ensure_ascii=False,indent=2); f.flush(); os.fsync(f.fileno())
            os.replace(temp,path)
        except Exception as exc:
            print(f"recovery failed for {path}: {exc}",file=sys.stderr,flush=True)

if __name__=="__main__":
    sys.stdin.buffer.read()
    recover(sys.argv[1])
