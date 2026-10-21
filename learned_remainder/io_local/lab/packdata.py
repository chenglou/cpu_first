"""The lab's data: a split as {"ids", "lab", "num"} tensors (via ../local_io.py's loader; IO_CACHE caches outside the
repository)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from local_io import load


def pack(split):
    ids, lab, num = load(split)
    return {"ids": ids, "lab": lab, "num": num}
