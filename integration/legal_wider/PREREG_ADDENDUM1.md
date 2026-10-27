# Phase D, addendum 1: ROUNDT's equivalence on minus1's own climb

Written 2026-10-27 15:14 PDT, after the file's tiny cores and before this arm's run. PREREG.md is unchanged.

## Why

PREREG.md predicted the file's tiny cores on H1, plusy, the cube, rho and minus1 to be phase C's unit for unit. They
are for H1, plusy and the cube (every unit the same in line and work). rho's differs only by `builds`' one comparison
in its reading unit (+825 work), its picks, library, rounds and answer programs the same. minus1's differs from its
level-3 pose: 115 candidates against phase C's 114 on the GPU. One of its laws had fewer than two computed rows at
the smallest size on the GPU's draws, so phase C dropped it; the file's empty target keeps it (PREREG.md, Design).
From there its climb is not phase C's: a round after level 6 as before, 13 rounds, not the same ones.

So minus1's climb does not show that the new ROUNDT keeps the same rounds. ROUNDT's verdicts were compared directly
(`checks/roundt_same_local.py`: 31 rounds, phase C's 13 of minus1 among them, the same verdicts, outputs and work),
but not on this climb.

## The arm

`research/submission_stage3_wider_no-roundt.py` (md5 `c764e681`, `make_wider.py --skip=roundt`): the file with phase
C's `round_t` (run_prog, T runs per count) in place of the round machine; every other function is the file's (an AST
comparison: only `SelfClimb.round_t` differs). One tiny core on minus1 (B200), `t_wider_oldroundt_minus1`.

## Prediction

| what | prediction | confidence |
|---|---|---|
| `t_wider_oldroundt_minus1` against `t_wider_minus1` | the same unit for unit: lines (seconds masked) and work, picks, library, rounds and answer programs | high |
| its round searches that keep rounds | slower than the file's (phase C's ROUNDT) | high |
