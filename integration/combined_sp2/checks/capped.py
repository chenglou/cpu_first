"""Runs a command with a wall-clock limit and a memory cap (the resident memory of its whole process group, polled every
2 s); past either, the group is killed. The command and its children die with this script.

    python capped.py SECONDS GIB -- COMMAND ...
"""
import os
import signal
import subprocess
import sys
import time

secs, gib = float(sys.argv[1]), float(sys.argv[2])
cmd = sys.argv[sys.argv.index("--") + 1:]
proc = subprocess.Popen(cmd, start_new_session=True)


def rss_gib(pgid):
    out = subprocess.run(["ps", "-o", "rss=", "-g", str(pgid)], capture_output=True, text=True).stdout
    return sum(int(x) for x in out.split() if x.isdigit()) / 2 ** 20


def kill(why=None):
    if why:
        print(f"CAPPED: {why}; killing the process group", flush=True)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):      # (already gone: macOS answers EPERM for a group of zombies)
        pass
    proc.wait()


t0, peak = time.time(), 0.0
try:
    while proc.poll() is None:
        time.sleep(2)
        peak = max(peak, rss_gib(proc.pid))
        if peak > gib:
            kill(f"{peak:.1f} GiB > {gib} GiB")
        elif time.time() - t0 > secs:
            kill(f"{time.time() - t0:.0f} s > {secs:.0f} s")
finally:
    if proc.poll() is None:
        kill()
print(f"EXIT {proc.returncode} after {time.time() - t0:.0f} s, peak {peak:.1f} GiB", flush=True)
sys.exit(proc.returncode or 0)
