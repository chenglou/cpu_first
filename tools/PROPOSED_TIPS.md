# Proposed additions to ~/github/tips-for-running-modal/AGENTS.md

Applied on 2026-10-28, with the lessons of the following week, in tips-for-running-modal e40e54e
(https://github.com/chenglou/tips-for-running-modal). Kept as the record of what this work added. These come from
building `modal_runner.py` here and testing it on Modal (client 1.2.6, 2026-10-22). Each bullet is written to
paste under the AGENTS.md section named in its heading.

## Launching Jobs

- **Re-attach a restarted driver to the calls it already spawned.** A launch with several workers keeps its one
  `spawn()` for a driver function that spawns the workers. Modal restarts a preempted driver with the same input.
  Without saved state, the restarted driver spawns the whole plan again, and the duplicates overwrite records and
  take container slots. So:
  - pass the driver a launch tag, and commit each spawned call's `object_id` to a Volume under that tag right after
    the spawn;
  - on start, re-attach with `modal.FunctionCall.from_id(id)` instead of spawning again;
  - make workers idempotent: return at once when their output is already on the Volume;
  - mark the driver `nonpreemptible=True` (CPU and memory at 3x list: cents for a 0.25-core driver).
  To test it, `modal container stop <driver container id>` hands the input to a new container, as a preemption does.
  Poll with `call.get(timeout=0)`: Python's built-in `TimeoutError` means the call is still running. After any other
  exception, check `call.get_call_graph()` before treating the call as finished. Cancelling the driver's call does not
  cancel the calls it spawned.
- **A memory request is not a limit.** `memory=16384` sets no limit at all. `memory=(16384, 65536)` adds a hard
  limit at 64 GiB. Billing is the higher of the request and the use.
- **Kill a subprocess's whole session.** `subprocess.run(timeout=...)` kills only the direct child. Its children (for
  example DataLoader workers) can live on in a reused container. Start long runs with `start_new_session=True` and
  SIGKILL the session (`os.killpg`) in a `finally`.
- **Pin GPU types that must not change.** `gpu="H100"` can land on an H200; `"H100!"` cannot. `gpu="A10G"` landed on
  an A10 once and an A10G once.

## Monitoring

- **Copy progress to the Volume for check-ins.** `modal container exec <id> -- ls /tmp` saw the worker's files on
  some containers and an empty `/tmp` on others (a CPU container and a T4 here). A worker that copies its log tail
  and memory use to the Volume every minute can be read at any time.
- **A container's Modal client has no CLI.** `python -m modal container list` fails inside a container (no
  `rich`). Call the client API instead, for example the `TaskList` RPC behind `modal container list`.
- **Measure cost per app.** `from modal.billing import workspace_billing_report` and then
  `workspace_billing_report(start=..., resolution="h")` give each app's cost per hour, complete hours only. To
  compare configurations, run each in its own app.

## Housekeeping

- This repo's `venv/bin/modal` points to `~/github/testing-modal/venv/bin/python3`, which no longer exists ("bad
  interpreter"). Recreate the venv, or use another environment's `python -m modal`.
