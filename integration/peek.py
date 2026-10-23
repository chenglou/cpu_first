"""Live progress of the integration runs: the last evaluator progress line of every running container of an app.

    cd ~/github/one-layer/one-layer-deeper && ~/github/sotaku/venv/bin/python ../cpu_first/integration/peek.py [APP]
"""
import json
import subprocess
import sys

PY = sys.executable
app = sys.argv[1] if len(sys.argv) > 1 else "integration"
out = subprocess.run([PY, "-m", "modal", "container", "list", "--json"], capture_output=True, text=True).stdout
for c in json.loads(out or "[]"):
    if (c.get("App Name") or "") != app:
        continue
    cid = c.get("Container ID")
    r = subprocess.run([PY, "-m", "modal", "container", "exec", cid, "--", "bash", "-c",
                        "for f in /tmp/*/stdout.log; do echo \"$f\"; grep -E 'step [0-9]+/|LR step=|RESULT|io_wiring|frozen|T_resolved' "
                        "\"$f\" | tail -2 | cut -c1-260; done 2>/dev/null"], capture_output=True, text=True, timeout=120)
    txt = r.stdout.strip()
    if txt and "/tmp/*/stdout.log" not in txt:
        print(cid, c.get("Start Time"), "\n ", txt.replace("\n", "\n  "), flush=True)
