"""Wait until the named files in the `integration` Volume's climb/ are newer than AFTER (epoch seconds), copy them to
integration/lib/records/ and print them (polls every 60 s, up to 6 h).

    python wait_vol.py AFTER NAME [NAME ...]          (NAME: a file name in climb/, e.g. profile_mul_3m.log)
"""
import sys
import time
from pathlib import Path

import modal

OUT = Path(__file__).resolve().parent / "lib" / "records"


def main(after, names):
    vol = modal.Volume.from_name("integration")
    pending, t0 = set(names), time.time()
    while pending and time.time() - t0 < 6 * 3600:
        try:
            have = {Path(e.path).name: e for e in vol.listdir("climb") if e.mtime > after}
        except Exception as exc:
            print("listdir failed:", repr(exc)[:200], flush=True)
            time.sleep(60)
            continue
        for name in sorted(pending & set(have)):
            data = b"".join(vol.read_file(have[name].path))
            (OUT / name).write_bytes(data)
            print(f"===== {name}\n{data.decode(errors='replace')[-6000:]}", flush=True)
            pending.discard(name)
        if pending:
            time.sleep(60)
    print("still pending:", sorted(pending), flush=True)


if __name__ == "__main__":
    main(float(sys.argv[1]), sys.argv[2:])
