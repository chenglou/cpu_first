"""The scale runs (SCALE.md), local or on Modal: bar3/breadth/drive.py's `fresh` and `carried`, on this folder's
learner (learn.py), through the breadth study's Tap (target jobs tagged, every lookahead kept)."""

from __future__ import annotations

from bar3.breadth import targets as TG
from bar3.breadth.drive import Tap
from bar3.scale import learn


def fresh(name, cfg, tname, search_many, emit, log=print):
    tap = Tap(search_many)
    TG.use(tname)
    tap.target = tname
    learn.run(name, cfg, tap.mapper, tap.emit_with_looks(lambda r: emit(dict(r, target_name=tname))), log=log)


def carried(name, cfg, order, search_many, emit, carry="targets", log=print, state=None, save=None):
    tap = Tap(search_many)

    def on_target(t):
        tap.target = t

    return learn.sequence(name, cfg, order, tap.mapper, tap.emit_with_looks(emit), carry=carry, log=log,
                          on_target=on_target, state=state, save=save)
