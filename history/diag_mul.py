import sys, runpy, torch
sys.argv = ["exp_wide.py", "--task", "mul", "--k", "4", "--seed", "0", "--steps", "3000", "--curriculum", "1"]
g = runpy.run_path("exp_wide.py", run_name="__main__")
ADD, acc_at = g["ADD"], g["acc_at"]
tt = ADD.truth_table()  # rows index (x, y, c) joint: x*20 + y*2 + c
bad = 0
for x in range(10):
    for y in range(10):
        for c in range(2):
            d, co = tt[x * 20 + y * 2 + c].tolist()
            s = x + y + c
            if (d, co) != (s % 10, s // 10): bad += 1
print("ADD wrong entries:", bad, "/ 200")
import exp_wide  # noqa
