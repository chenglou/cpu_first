"""Extract (lib, progs) from a stage-3 record's round_found line into a JSON {lib: {name: text}, round: [text]}."""
import ast, json, re, sys
rec, out = sys.argv[1], sys.argv[2]
r = json.load(open(rec))
for l in r["stdout"].splitlines():
    if l.startswith("LR round_found"):
        m = re.match(r"LR round_found step=(\d+) programs: (\[.*?\]) library: (\{.*\})\s*$", l)
        progs = ast.literal_eval(m.group(2)); lib = ast.literal_eval(m.group(3))
        json.dump({"step": int(m.group(1)), "lib": lib, "round": progs}, open(out, "w"), indent=1)
        print(out, "step", m.group(1), len(lib), "entries", len(progs), "programs")
        break
