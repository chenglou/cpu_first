# Handover: the hygiene pass before this repo leaves the laptop

*Written 2026-10-23, with [REVIEW.md](REVIEW.md), for `main` at `f55fdca`. Brought up to date on 2026-10-24, once the
later studies were committed: §1's scan covers their commits, and §5's sizes include them. The same day the repo went
to a private GitHub repository (§6). Early on 2026-10-25, after the reruns and bar 3's new studies landed, §1's scan
was extended to their commits and §4 and §5's sizes were remeasured.*

## 1. Secrets and private data

**What was scanned:** every blob reachable from any ref (all 41 commits of `main`: 3,475 blobs, 310 MB uncompressed;
there are no other branches, tags, stashes, remotes or unreachable objects), and the working tree.

**What it looked for:**
- keys and tokens: AWS, GitHub, OpenAI and Anthropic, Slack, Google, Hugging Face, Modal, the competition CLI's own
  API keys, bearer tokens, private-key blocks, `password = "…"`-style assignments, token variable names, and URLs
  with credentials;
- credential-like file names (`.env`, `*.pem`, `id_rsa`, `.netrc`, `.modal.toml`, …);
- email addresses;
- home-folder paths, `~/github/…` paths, the Claude app's scratch folders, and Modal app URLs.

**Again on 2026-10-24:** every line added by the six commits since `f55fdca` (220 files: this pass, the ablation,
self-practice, the shader follow-up and the doc updates), for the same patterns. No credentials, no emails, and one
home path, in a script default (`integration/self_practice/core.py`, now fixed; see the table).

**Again on 2026-10-25:** every line added by the 13 commits since `6596beb` (313 files: the reruns after the spend
limit, the weak-check fix in eight certified variants, bar 3's finished runs and its subgoal study, and the doc
updates), for the same patterns and for the competition key's exact value. No credentials and no emails; the author's
home path only inside four NCA relaunch logs (`nca/logs/rest1_relaunch*.log`), left as evidence like the others.

| finding | where | done |
|---|---|---|
| **credentials** | nowhere, in the tree or in the history | nothing to do |
| **emails** | only the commit author's (one author and committer on all 41 commits) | nothing to do |
| Claude scratch-folder paths in docs | `panel/BRIEF.md`, `panel/critiques.md`, `panel/designs.md`, `panel_synthesis.md` (14 places) | **fixed**: now `$SCRATCH`, with a note that it was the author's scratch folder and its files are not in this repo |
| the same paths as a script's default H1 folder | `h1.py`, `anchors/core.py`, `anchors/modal_lab.py`, `ablation/probe_trap.py`, `microlab/ext_loop_csub_modal.py`, `learned_remainder/{adder,chain,control,io,square}_local/*.py` | **fixed**: the default is now `../one-layer-deeper/data/generated/<the H1 folder>`, relative to the script; `H1_DATA` still overrides it. The data is the same: all 18 data files are byte-identical, and only `dataset_config.json`'s `output_dir` differs |
| the author's home path as a default H1 folder | `integration/lib/cells.py`, `integration/lib/inclimb.py`, `library/inrun.py`, `modal_relaxed.py`, `h1_local_cpu.json`, `h1_local_mps.json` | **fixed** the same way. The two local manifests now hold the official relative `data_root`, so they run from `../one-layer-deeper` like the official manifest (README.md says so) |
| the author's home path as a default H1 folder, in a file committed after this pass | `integration/self_practice/core.py` | **fixed** the same way (2026-10-24) |
| a link into another project of the author's | `handwired_bridge/HANDWIRED_BRIDGE.md` | **fixed** (§2) |
| Claude scratch-folder paths, the author's home path, and the author's Modal workspace name (in Modal app URLs) | inside run logs and records: about 220 files under `runs/`, `*/records/`, `*/runs/`, `*/logs/`, `microlab/runs_*/`, two of them scripts kept with their outputs in `runs/` | **left**: they are evidence, and the records are not to be edited. They grant no access |
| `~/github/<project>` paths, mostly `~/github/sotaku/venv/bin/python` (the author's interpreter) | the documented commands of 33 docs and 103 scripts | **left**: REVIEW.md tells a reviewer to use any Python with torch, numpy and Modal 1.2.6. Rewriting 136 files would touch other sessions' write-ups for no safety gain |
| older versions of the files fixed above | git history only | **needs a history rewrite** to remove; see below |

**Should the history be rewritten?** Not for safety: no secret is in it. What is in it is the author's home path,
local project names, Claude scratch-folder paths and the Modal workspace name. A rewrite (`git filter-repo
--replace-text`, on a fresh clone) would change every commit hash, and the write-ups cite commit hashes
(OVERVIEW §3, §5). It would also change the records' logs. A snapshot without history (§5, option 4) drops the
history-only paths but keeps those in today's files, and loses the commits the docs cite. Recommendation: hand over
the history as it is.

## 2. Links

All 57 tracked Markdown files, plus REVIEW.md and this file, were checked. The check covers every relative link,
every heading anchor (GitHub's slug rules), and whether each target is tracked: 350 relative links, 4 with anchors,
before REVIEW.md.
- One was broken: `handwired_bridge/HANDWIRED_BRIDGE.md` linked a tutorial in the separate handwired project, outside
  this repo. It now names the file in plain text.
- Nine links (eight in the older docs, one in REVIEW.md) point into `../one-layer-deeper/`: its README and its
  `#rules` anchor. They resolve once the competition clone sits next to this folder, as REVIEW.md asks.
- No link points at an untracked file, except the new links to REVIEW.md and HANDOVER.md themselves, until they are
  committed.
- **Again on 2026-10-24**, with the later studies committed: 61 tracked Markdown files, 451 relative links, none
  broken; the same 9 point into `../one-layer-deeper/`.

## 3. Checksums

- `python3 tools/verify_records.py`: 10 of 10 records check out. These are the official records of s3i
  (`0e813144…`), stage 1 (`cfd7b61d…`) and general v3 (`0805b7d3…`), and s3i's seeds 1–7. For each one:
  - the embedded source's md5 equals the file's;
  - the manifest is the competition's, or differs only in its seed;
  - the `RESULT_JSON=` line equals the result;
  - T = 64 on both ladders, every rung 768/768, every split exact.
- s3i is rebuilt byte for byte from its sources by `integration/make_stage3.py`; `make_stage3.py s3h` confirms s3h's
  build too.
- The local H1 folder hashes, the way the runner hashes it, to `a6414ec9fc6a5efce6414895b52d4eb7`: the value in every
  shared-runner record.
- The competition's static source check (`submission_validation.validate_submission_source`, as `submission.py`,
  256 KiB) passes for all three files.

## 4. Junk

**Trashed** with `trash`, so they can be recovered from the macOS Trash: 12 untracked `__pycache__` folders, about
1 MB. There were no `.DS_Store` files, temp files or stray untracked outputs.

**Kept on purpose, on disk only.** `.gitignore` keeps 17 run outputs over 5 MB out of git: search dumps and two
checkpoints, 337 MB in all on 2026-10-25, the largest 54 MB. Only two are cited by name, the two `ladder.jsonl` files
that HANDWIRED_BRIDGE.md mentions, and a clone does not have them.

**Not touched:** the work of the sessions still running then: `nca/` (uncommitted changes),
`integration/self_practice/`, `integration/SELF_PRACTICE.md`, `integration/ablation_s3/` and
`integration/ABLATION_S3.md` (untracked then). All of it has since been committed.

**Three records over 5 MB are tracked on purpose:** `nca/runs/trim1.jsonl` (14.6 MB), the records of NCA.md §9's 64
runs from random weights, and `nca/runs/rest_stab.jsonl` (8.8 MB) and `nca/runs/rest1.jsonl` (6.4 MB), two of §10's.
The sections' numbers are checked against them.

### Proposed removals (not done: the user chose to keep every log)

1. **38 waiter transcripts**, 172 KB. These are the polling logs of the per-folder fetch waiters. No doc cites them,
   and the records hold what they waited for.

   ```bash
   trash general_lessons/logs/wait_wave*.log litmus/logs/wait_*.log wider_menus/logs/wait_*.log runs/learned_wiring/wait_*.out runs/wider_menus/wait_*.out
   ```

2. **31 `.partial` logs that have a finished twin.** A diagnostic's live log was saved as `<name>.partial`, and its
   final log as `<name>.log` beside it. Each of these 31 partials is wholly contained in its `.log`. The other 68
   `.partial` files are the only log of their run, so keep them.

   ```bash
   for f in $(git ls-files '*.partial'); do b="${f%.partial}"; for e in "" .log .txt .json .out; do [ -e "$b$e" ] && echo "$f" && break; done; done | xargs trash
   ```

To stage the deletions without staging the running sessions' changes, name the folders:
`git add -u general_lessons/logs litmus/logs wider_menus/logs runs/learned_wiring runs/wider_menus` for the first, and
`git add -u learned_remainder/combined_local/logs learned_wiring/local/logs wider_menus/local/logs` for the second.

## 5. What goes, and how

**Tracked:** 4,698 files at `main` early on 2026-10-25 (4,409 the afternoon before, 3,635 that morning), 647 MB as
files. The largest is `nca/runs/trim1.jsonl`, 14.6 MB (§4), then `nca/runs/rest_stab.jsonl`, 8.8 MB, and
`nca/runs/rest1.jsonl`, 6.4 MB; every other tracked file is under 5 MB, and all are far under GitHub's 100 MB per-file
limit (and its 50 MB warning). Packed, the whole history is about 55 MB.

**On disk only** (in no clone, bundle or push):
- the 17 ignored outputs (337 MB);
- the running sessions' untracked files (early on 2026-10-25, the bridge session's carry study, in progress);
- every uncommitted change.

**Commit first:** anything still uncommitted. A clone, bundle or push carries only what is committed.

**The competition repo** goes separately. The reviewer clones it next to this folder, at the commit the records used:

```bash
git clone https://github.com/tilde-research/one-layer-deeper.git && git -C one-layer-deeper checkout 87e5c26
```

**Options** (sizes remeasured early on 2026-10-25 from test copies, since trashed; option 3 is the one the user
chose, §6):

1. **A git bundle** (recommended): one file of about 55 MB, the full history, only what is committed.

   ```bash
   cd ~/github/one-layer/cpu_first
   git bundle create ~/Desktop/cpu_first.bundle HEAD main
   git bundle verify ~/Desktop/cpu_first.bundle
   # the reviewer: git clone cpu_first.bundle cpu_first
   ```

2. **A zip of a fresh clone**: about 175 MB (the snapshot plus the bundle; not built this time), the files plus a
   packed `.git`. `--no-local` packs the objects instead of hard-linking them. Removing `origin` drops the laptop's
   path from `.git/config`; the clone's reflog still notes where it came from.

   ```bash
   cd ~/Desktop && git clone --no-local ~/github/one-layer/cpu_first cpu_first && git -C cpu_first remote remove origin
   zip -qry cpu_first.zip cpu_first
   ```

3. **A private GitHub repository**: the push is about 55 MB, and the largest tracked file is 14.6 MB. This uploads to a
   third-party service.

   ```bash
   cd ~/github/one-layer/cpu_first
   gh repo create cpu_first --private --source=. --remote=origin --push
   gh api -X PUT repos/<your-login>/cpu_first/collaborators/<reviewer-login> -f permission=pull
   ```

4. **A snapshot without history**: about 118 MB. No commit hash the docs cite can be checked from it.

   ```bash
   git -C ~/github/one-layer/cpu_first archive --format=zip --prefix=cpu_first/ -o ~/Desktop/cpu_first_snapshot.zip HEAD
   ```

To include the on-disk-only outputs as well (337 MB before compression):

```bash
cd ~/github/one-layer/cpu_first && git ls-files --others --ignored --exclude-standard -z | xargs -0 tar czf ~/Desktop/cpu_first_big_outputs.tgz
```

## 6. The user's decisions (2026-10-24)

- The logs' home path, local project names and Modal workspace name are fine to share. The logs stay as they are, and
  the history is not rewritten.
- The proposed removals (§4) are not made: every log is kept.
- The handover is option 3: a private GitHub repository, [`chenglou/cpu_first`](https://github.com/chenglou/cpu_first),
  created with `gh repo create cpu_first --private --source=. --remote=origin --push`. The user adds the reviewer as a
  collaborator.
