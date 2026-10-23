"""Diagnostic copies of a stage-3 file (INTEGRATION.md §3, s3h): the file with three switches, all off by default, for
research runs only (never certified, never submitted).
  * REPLAY {step: {"lib": {name: text}, "progs": [text]}}: the climb is not run; at that step the recorded run's
    library and round programs are set as its climb set them (phase D). The climb uses its own generators, never the
    global one, and changes no parameter, so the training before and after is the recorded run's;
  * DUMP_STEPS (steps): at each, the round's answer pool (N, x, y as read, as integers) is printed (POOL_DUMP);
  * GRAD_CHECK: every parameter's gradient is checked in phase D (GRAD_BAD: the first steps with a non-finite one), and
    T's direction lesson prints its weights (T_STATS).

    python integration/lib/debug_patch.py SRC OUT
"""
import ast
import sys

src, out = sys.argv[1], sys.argv[2]
s = open(src).read()


def sub(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:80])
    s = s.replace(old, new)


sub('LOG_EVERY = 50\n', 'LOG_EVERY = 50\nREPLAY = {}\nDUMP_STEPS = ()\nGRAD_CHECK = False\n')
sub('''    def climb_unit(self, Nd: Tensor, Xd: Tensor) -> None:
        """One unit of the climb (phase C): the read rows join the pool, then the next search or sleep."""
        ctl = self.ctl
''', '''    def climb_unit(self, Nd: Tensor, Xd: Tensor) -> None:
        """One unit of the climb (phase C): the read rows join the pool, then the next search or sleep."""
        ctl = self.ctl
        if ctl.step in DUMP_STEPS:
            v = lambda d: [int("".join(str(int(t)) for t in reversed(r)) or 0) for r in d.tolist()]
            if ctl.ans_pool:
                Nd_, Xd_, Yd_ = (torch.cat([a[i] for a in ctl.ans_pool]) for i in range(3))
                print(f"POOL_DUMP step={ctl.step} rows={[list(t) for t in zip(v(Nd_), v(Xd_), v(Yd_))]}", flush=True)
            print(f"CELLS_DUMP step={ctl.step} {self.cells_right()}", flush=True)
        if REPLAY:
            if ctl.cells is None:
                ctl.cells = self.snapped_cells()
                print(f"LR climb_starts step={ctl.step} (replay) {self.cells_right()}", flush=True)
            if ctl.step in REPLAY and not ctl.progs:
                r = REPLAY[ctl.step]
                ctl.lib = {k: parse(v) for k, v in r["lib"].items()}
                ctl.progs = [parse(p) for p in r["progs"]]
                ctl.phase, ctl.in_phase = "D", 0
                ctl.history.append((ctl.step, "round", len(ctl.progs)))
                print(f"LR round_found step={ctl.step} (replay) programs: {[show(p) for p in ctl.progs]}", flush=True)
            return
''')
if "        hit = (Lpl + Lmi) > 0\n" in s:                         # s3g's T lesson (weights)
    sub('''        hit = (Lpl + Lmi) > 0
''', '''        hit = (Lpl + Lmi) > 0
        if GRAD_CHECK:
            print(f"T_STATS step={ctl.step} n={n} hits={int(hit.sum())} plus_right={float(Lpl.mean()):.3f} "
                  f"minus_right={float(Lmi.mean()):.3f} wp=[{float(wp.min()):.3e},{float(wp.max()):.3e}] "
                  f"wm=[{float(wm.min()):.3e},{float(wm.max()):.3e}]", flush=True)
''')
else:                                                                # s3h's (log weights)
    sub('''        hit = good.any(1)
''', '''        hit = good.any(1)
        if GRAD_CHECK:
            print(f"T_STATS step={ctl.step} n={n} hits={int(hit.sum())} plus_right={float(Lpl.mean()):.3f} "
                  f"minus_right={float(Lmi.mean()):.3f} log_wp=[{float(lw[:, 0].min()):.3e},{float(lw[:, 0].max()):.3e}] "
                  f"log_wm=[{float(lw[:, 1].min()):.3e},{float(lw[:, 1].max()):.3e}]", flush=True)
''')
sub('''    def control(self, aux: dict, loss: float) -> None:
        """The schedule, once per step (after the label lessons): the reader's stages, the phases, the freeze."""
        ctl = self.ctl
''', '''    def control(self, aux: dict, loss: float) -> None:
        """The schedule, once per step (after the label lessons): the reader's stages, the phases, the freeze."""
        ctl = self.ctl
        if GRAD_CHECK and not getattr(ctl, "hooked", False):
            ctl.hooked, ctl.bad = True, 0
            for name, p in self.named_parameters():
                def hook(g, name=name):
                    if ctl.phase == "D" and not bool(torch.isfinite(g).all()) and ctl.bad < 40:
                        ctl.bad += 1
                        print(f"GRAD_BAD step={ctl.step} param={name} nonfinite={int((~torch.isfinite(g)).sum())} "
                              f"absmax_finite={float(g[torch.isfinite(g)].abs().max()) if bool(torch.isfinite(g).any()) else 0.0:.3e}",
                              flush=True)
                    return g
                p.register_hook(hook)
        if GRAD_CHECK and ctl.phase == "D":
            print(f"LOSS_D step={ctl.step} loss={loss:.6g}", flush=True)
''')
ast.parse(s)
open(out, "w").write(s)
print("wrote", out)
