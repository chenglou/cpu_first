# Composition round 7, addendum 1: general fixes for the one failure, pre-registered

Written 2026-10-27, after the pre-registration's eight arms and before any run below. FAMILY.md §1 records its time
and hash.

## What the eight arms showed

**Six of the seven tasks came,** each right on both of H1's ladders (14 of 14 rungs at 768/768), and on every one of
the ladders' 747 moduli at x = 0, at a random x and at x = N − 1:

| task | the round | how |
|---|---|---|
| x² (H1) | level 2 (569 s) | x², then the reduction, as in round 6 |
| the cube | level 3 (1,576 s) | two calls of the window mod(mul(a,b),N), itself built over the window mul(mod(a,N),b) |
| ten | level 2 (1,192 s) | x·10 (the constant's operation, `LDA.X SHL.A`), then the multiply and the reduction |
| rho | level 3 (1,728 s) | the window add(mul(a,b),1), then the reduction |
| plus3 | level 4 (2,259 s) | the window add(mul(a,b),3), then the reduction. Make 3 came at level 2, 4 tokens over the link (`CLA B[T.red.pair.AX ADC.S]`) |
| tenrho | level 3 (1,712 s) | the window mul(mod(mul(a,b),N),10), then `ADC.X`, then Horner's step. The `ADC.X` adds the 1 that a finished reduction's last comparison leaves as its borrow |

The replay gave round 6's record again, apart from times.

**minus1 did not** (the deadline in level 6). Its reading, mod(pred(add(mul(X,X),N)),N), has two problems:
- **No 1.** It has no constant leaf, so nothing makes a 1, and pred's subtraction needs one.
- **Holes below N.** A window's hole is drawn below N, but the route puts x² + N in it. The window mod(pred(a),N),
  learned at level 5 on holes below N, has no full reduction, so the two-call route was wrong at full size. The check
  shows the cause (`logs/family_check.txt`): on holes below N, one conditional subtraction passes for the reduction of
  a + N − 1.

## The fixes, each behind its own flag

1. **CONST1:** succ and pred are an addition and a subtraction of 1, a constant the reading holds without a leaf.
   When the reading has one, its 1 is made from N (const.1 on n), and succ or pred is posed on x,n. Their answers are
   the law reader's own: x + 1, and max(x − 1, 0). It needs CONST's patterns.
2. **HOLES:** a window's holes are drawn from other rows than its modulus's (w1o: another row's x; w2o: two other
   rows'). A hole is then above N as often as not, as in the routes. The windows' bindings stay as before.

Both name no task. CONST1 poses nothing for a reading without succ or pred, so it cannot change x², the cube, ten,
rho, plus3 or tenrho.

## Arms

Each on one B200, the pre-registration's setup (SECONDS 3,600), launched together, at most 8 at once:
- `r7_minus1_both`: minus1 with both fixes;
- `r7_minus1_const1` and `r7_minus1_holes`: minus1 with each alone;
- the family with both fixes: `r7_x2_both`, `r7_x3_both`, `r7_ten_both`, `r7_rho_both`, `r7_plus3_both` and
  `r7_tenrho_both`. Since CONST1 poses nothing without succ or pred, the x² and cube arms here are HOLES's replays;
- `r7_x2_const1` and `r7_x3_const1`: CONST1's replays.

Any round is checked on both ladders to T = 64.

## Predictions

| what | prediction | confidence |
|---|---|---|
| r7_minus1_both | the round by level 8, right on both ladders: make 1, pred, the windows over them, then the multiply and the window | low-medium |
| r7_minus1_const1 | no round: on holes below N the windows reduce with one conditional subtraction, wrong at the route's values | medium |
| r7_minus1_holes | no round: nothing makes a 1 | medium-high |
| r7_x2_both | the round by level 3 | high |
| r7_x3_both | the round by level 4, two calls of the window | medium-high |
| r7_ten_both | the round by level 3 | high |
| r7_rho_both, r7_plus3_both, r7_tenrho_both | each round again, by level 6 | medium |
| r7_x2_const1, r7_x3_const1 | r7_x2's and r7_x3's records again, apart from times | high |

## Money

- Eleven arms, at most 8 at once. The watcher, over both apps, stops all at $79 (the cap less one hour of 8 B200s).
- This study so far: about $31. With these, about $73, under the soft $90.
