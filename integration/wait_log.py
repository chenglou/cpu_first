"""Wait until a running integration run's live log (in its container) shows a pattern, or the run ends; prints the
matching lines. Polls every 120 s, up to 8 h.

    cd ~/github/one-layer/one-layer-deeper && python ../cpu_first/integration/wait_log.py RUN 'LR round_found|LR frozen'
"""
import json
import re
import subprocess
import sys
import time

PY = sys.executable
run, pat = sys.argv[1], re.compile(sys.argv[2])
t0 = time.time()
while time.time() - t0 < 8 * 3600:
    out = subprocess.run([PY, "-m", "modal", "container", "list", "--json"], capture_output=True, text=True).stdout
    alive = False
    for c in json.loads(out or "[]"):
        if c.get("App Name") != "integration":
            continue
        r = subprocess.run([PY, "-m", "modal", "container", "exec", c["Container ID"], "--", "bash", "-c",
                            f"test -f /tmp/{run}/stdout.log && cat /tmp/{run}/stdout.log"],
                           capture_output=True, text=True, timeout=300)
        if r.returncode == 0 and r.stdout:
            alive = True
            hits = [l for l in r.stdout.splitlines() if pat.search(l)]
            if hits:
                for l in hits[-5:]:
                    print(l[:1500], flush=True)
                sys.exit(0)
            climb = [l for l in r.stdout.splitlines() if l.startswith("LR climb")]
            print(time.strftime("%H:%M"), climb[-1][:200] if climb else "(no climb yet)", flush=True)
    if not alive:
        print("run not found in any container (finished or not started)", flush=True)
        sys.exit(1)
    time.sleep(120)
