# cpu_first — experiments for a CPU-first take on One Layer Deeper

Everything runs on CPU in seconds to minutes. Data: the released H1 set, generated locally with the competition
repo (`python -m data.reversed_n_squaring_mod ...`, the H1 command in `scripts/generate_datasets.sh`). Point
`H1_DATA` at that folder if it moves. Python: torch + numpy.

| file | what it shows |
|---|---|
| `REPORT.md` | the write-up |
| `FINDINGS_substrate.md` | raw numbers for every experiment below |
| `h1.py` | H1 loader in LSB-first digit form + an evaluator-faithful certifier (T-ladder, both profiles) |
| `cells.py` | learned truth-table cell (soft / straight-through / hard) — the "gate" of a learned machine |
| `exp_substrate.py`, `exp_codes.py`, `exp_wide.py` | answers-only learning of composed digit tables: single table easy; latent 10-way wires stall; wide (24–48-way) wires fix shallow compositions; deep (multiplication) compositions stall even without any modular reduction |
| `machine.py` | decimal Montgomery machine (panel CADET-M datapath): Peano self-test ALU + vectorised microcode |
| `cadet.py` | end to end: self-tests → label speculation for N′ → output map → T=64 certified on both profiles |
| `menu.py` | unknown recurrence: labels select among 24 complete programs; 6/6 hidden recurrences certified |
| `speculate_gate.py` | the binary gate from labels only: 65,536 candidate machines bitsliced, 4 rows → the full adder |
| `speculate_wiring.py` | 31,104 wirings at once; factored search via the label-free round-trip (finds a base-12 impostor) |
| `speculate_full.py` | gate × wiring (≈2³¹) via round-trip then labels → exactly full adder + reference wiring |
| `alias_width_audit.py` | the same netlist at 24 bits passes every training row and the training-pool profile but fails 2,068/5,376 OOD rows: speculation cannot see off-distribution structure |
| `panel/`, `panel_synthesis.md` | the design panel: brief, prior-art survey, 5 designs, 5 adversarial critiques, synthesis |
| `bitcpu_netlist.py` | panel critic's gate-level BitCPU (every gate = one FA table): `python bitcpu_netlist.py $H1_DATA 01101001,00010111` certifies T=64 on both profiles |
| `submission.py` | SpecBit as a competition submission: that netlist in torch, its one cell learned inside the forward pass as a mixture over all 65,536 truth tables (bitsliced, 64 per int64 word). Certifies T=64 on both profiles through the unmodified `benchmark.runner` |
| `h1_local_mps.json`, `h1_local_cpu.json` | local copies of `h100_hard_h1.json` (device `mps:0` / `cpu`, fp32, amp off, `num_workers` 0, `log_every` 1); edit `data_root` to your generated H1 folder |
| `spec_grammar.py` | bitsliced speculation in torch over a grown grammar of whole machines (gate, converters, entry/exit, a two-operand Montgomery core with radix-2 and radix-4 families): staged search with label-free round trips, label-free core laws (stage C0, `--law`), labels, every training row, then the certification audit. `--families r2 --ppop ab --law none --kw-max 28` reproduces the first 2^54.4 runs; `--core ports` adds carry-save port sources; `--law-synth`, `--count-law` and `--c-check decimal` make the search label-free (REPORT §10); `--cell-law` and `--width-law` add per-cell value laws and a width bound inside the core, and `--cell-audit` checks them (REPORT §11); `--kw-max` (default 32) sets the largest K, W and word width, `--digit-scales` the value laws' per-row scales, `--witness` adds declared-range witness rows to the audit (REPORT §12); `--collapse` runs D and E on one machine per class of law-equivalent survivors (`--collapse C1` already groups stage C's survivors, so the semigroup check also runs once per class), and `--collapse-check` spot-checks other members of each class (REPORT §13–14). `--bench` measures raw throughput, `--fuse-probe` the `torch.compile` headroom of the hot block, `--alias-demo` the T=1 aliases without the entry/exit round trip |
| `modal_h100.py` | the same on a Modal H100, in an image mirroring the competition's `modal_runner.py`: `--task h1` runs the official H1 manifest untouched, `--task bench` / `--task search` run `spec_grammar.py` |
| `RELAXED.md` | which single relaxed rule makes the step learnable by a mostly generic network: per-step labels, show-your-work traces (and curricula), periodic units tied to N in fp64, unlimited data, and supervising s². A trace GPT whose every token is local (trial subtractions) certifies T=64 up to 24 bits |
| `relax_common.py`, `relax_step.py`, `relax_trace.py`, `relax_trace_errors.py`, `relax_periodic.py`, `relax_periodic_cert.py`, `relax_square.py`, `relax_table.py` | the RELAXED.md experiments; `relax_table.py` prints every result |
| `modal_relaxed.py` | runs any `relax_*.py` on a Modal H100 (Volume `relaxed-runs`; copies in `runs/modal/`) |
| `runs/` | logs and JSON results of every run above (MPS, CPU, H100) |
