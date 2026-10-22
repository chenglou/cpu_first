"""The compiled handwired tick as a trainable PyTorch network, loosened notch by notch.

`load.Program` runs handwired's compiled layers as they are. This module reads the same compiled program and regroups
its weights by what they do, so that each part can be freed on its own:

- **cell circuits.** Every scratch cell of a kind does the same work on its three ports (handwired builds one copy per
  cell; `extract` checks that every copy is the same). A kind's circuit is a list of units in layers, each unit either
  affine, `left`, or ReGLU, `left * relu(gate)`, where `left` and `gate` are affine forms over the cell's ports and the
  units of earlier layers. Its weights, read off the compiled operations:
      SUB    9 products a_i * relu(b_j), 18 products (a_i b_j) * relu(c_k), 5 output rows (the table: 0/1 weights)
      SEL    1 - c_1, the products c_1 * relu(a_i) and (1 - c_1) * relu(b_i), their sums (emitSelect), constants
      SHIFT  the copies a_i, constants
- **the routing.** Which cells a port reads. Compiled, port h of cell p reads exactly cell comp[h, p]: in handwired's
  layers that is the sparsity pattern of the first layer's weights (`load.Program`); here it is a separate matrix
  R_h [L, L], one-hot rows, so that the pattern itself can be loosened.
- **the loop.** The same block every tick (tied), as the host loop runs the same program.

Notches (`HCfg`):
    tables   "compiled" | "learned": the SUB output rows, from random; `tparam` "simplex" (each joint unit's row a
             softmax over digits and over borrows, the attention bridge's table form), "raw" (plain affine weights) or
             "densetab" (the rows a dense ReGLU network's outputs on each joint unit's one-hot input: the H1 form)
    gates    "compiled" | "learned": every unit's values in the freed kinds' circuits (`free_kinds`; products, select,
             copies, constants, the table too) from random, on the compiled sparsity pattern; `lift` re-expresses the
             circuit through a random projection of each port's state (the user's projection) | "dense": each freed
             kind's cell a generic ReGLU network over its ports (`DenseCell`), the cell's own wiring freed
    route    "fixed" | "menu": per (port, kind) a multiplexer over a menu of relative sources, port = sum_o w_o * S[src_o]
             | "dense": per port a map over every earlier cell, weights from a joint table over (query kind, key kind,
             row offset, column offset, top place)
    mux      how a menu or dense port weighs its sources from its gates (HNet.mux): "softmax" (the attention bridge's) |
             "relu" (handwired's multiplexer, normalized) | "relu_free", "sigmoid_free" (no competition in the
             gradient) | "raw" (unnormalized) | "softplus" (normalized, a gate below 0 keeps a gradient)
    tied     one block for every tick, or one per tick (untied)
    copies   Net2Net-style widening: every compiled unit, or every port's gates, in this many copies
States are raw values; what the host writes back after a soft tick is `writeback` ("raw", "norm" or "softmax" per
digit group), and the lessons read a digit group the same way (`train.py`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bridge import grid as G
from bridge import net as BN
from bridge.grid import NIN, SCRATCH, SEL, SHIFT, SUB, VIN, Z

from handwired_bridge import load

NK = len(G.KINDS)


# ---------------------------------------------------------------- the compiled cell circuits

@dataclass
class Unit:
    kind: str                     # "affine" or "reglu"
    left: dict                    # z index -> coefficient
    lbias: float
    gate: dict
    gbias: float
    layer: int


def extract(rec):
    """Every kind's cell circuit from a compiled record: {kind: (units, outputs)}, where a unit's z indices count the
    cell's port entries first (port h, entry e at h * (B + 2) + e; entries are B digit bits then 2 borrow bits) and
    then the circuit's own units, and `outputs` lists the units that hold the next state's B + 2 entries. Checks that
    every cell of a kind has the same circuit."""
    lay = rec["layout"]
    B = lay["B"]
    S = B + 2
    where = {}                                                         # source wire -> (cell, entry)
    for p, cell in enumerate(rec["grid"]):
        for e, wire in enumerate(cell["digit"] + cell["borrow"]):
            where[wire] = (p, e)
    ops = {o["wire"]: o for o in rec["operations"]}
    nxt = rec["nextSource"]
    circuits = {}
    for p in range(lay["L"]):
        kd = lay["kind"][p]
        if kd not in SCRATCH:
            continue
        ports = [lay["ports"][h][p] for h in range(3)]
        outs = [nxt[w] for w in rec["grid"][p]["digit"] + rec["grid"][p]["borrow"]]
        units, index = [], {}                                        # op wire -> unit position

        def z_of(wire):
            if wire in where:
                q, e = where[wire]
                hs = [h for h in range(3) if ports[h] == q]
                assert hs, f"cell {p} reads cell {q}, which is none of its ports"
                # a cell read by two ports (the top SEL cell reads one cell as B and as C): a borrow through port C
                # when C reads it, a digit through the first other port
                if len(hs) > 1:
                    hs = [2] if e >= B and 2 in hs else [h for h in hs if h != 2] or hs
                return hs[0] * S + e
            return 3 * S + visit(wire)

        def visit(wire):
            if wire in index:
                return index[wire]
            o = ops[wire]
            left = {}
            for w2, c in o["left"]:
                zi = z_of(w2)
                left[zi] = left.get(zi, 0.0) + c
            gate = {}
            for w2, c in o["gate"]:
                zi = z_of(w2)
                gate[zi] = gate.get(zi, 0.0) + c
            deps = [i - 3 * S for i in list(left) + list(gate) if i >= 3 * S]
            layer = 1 + max([units[d].layer for d in deps], default=0)
            units.append(Unit(o["kind"], left, o["leftBias"], gate, o["gateBias"], layer))
            index[wire] = len(units) - 1
            return index[wire]

        out_units = [visit(w) for w in outs]
        # a canonical order: by layer, then by first appearance
        order = sorted(range(len(units)), key=lambda u: (units[u].layer, u))
        new = {u: i for i, u in enumerate(order)}
        remap = lambda d: {(k if k < 3 * S else 3 * S + new[k - 3 * S]): v for k, v in d.items()}
        canon = [Unit(units[u].kind, remap(units[u].left), units[u].lbias, remap(units[u].gate), units[u].gbias,
                      units[u].layer) for u in order]
        circ = (canon, [new[u] for u in out_units])
        if kd in circuits:
            assert same_circuit(circuits[kd], circ), f"cell {p} of kind {G.KINDS[kd]} differs from the first one"
        else:
            circuits[kd] = circ
    return circuits


def same_circuit(a, b):
    (ua, oa), (ub, ob) = a, b
    if oa != ob or len(ua) != len(ub):
        return False
    return all(x.kind == y.kind and x.left == y.left and x.gate == y.gate and x.lbias == y.lbias and
               x.gbias == y.gbias and x.layer == y.layer for x, y in zip(ua, ub))


# ---------------------------------------------------------------- the network

@dataclass
class HCfg:
    tables: str = "compiled"
    tparam: str = "simplex"
    gates: str = "compiled"
    route: str = "fixed"
    mux: str = "relu"             # see HNet.mux
    menu: str = "wide"             # menu options: "wide" (the attention bridge's 7) | "rel" (+ menu_extra offsets)
    menu_extra: int = -1
    tied: bool = True
    nblocks: int = 0
    init: float = 0.1              # std of random weights
    gate_init: float = 0.0         # menu/dense gates start at this value plus noise of std `init` (0: 1 / options)
    window: tuple = (3, 2)         # dense routing: rows up to Rb above (bin -Rb also holds every row further up),
                                   # columns within +-Cb (clipped bins)
    causal: bool = True            # dense routing: only cells before the cell in raster order
    hidden_mult: int = 1           # gates "dense": units per layer = compiled units * hidden_mult
    writeback: str = "raw"         # what the host writes back after a soft tick: "raw" values; "norm": each digit
                                   # group clamped at 0 and normalized; "softmax": softmax(wb_beta * values) per
                                   # group (the soft counterparts of the hard path's snap)
    wb_beta: float = 10.0
    free_kinds: tuple = ("SHIFT", "SUB", "SEL")   # gates "learned" / "dense": the kinds whose cells are freed
    dense_inputs: str = "full"     # gates "dense": "full" ports or the "used" entries (see DenseCell)
    dense_out: bool = False        # gates "learned": the output rows read every joint product (the table learned,
                                   # not given by the compiled sparsity)
    lift: int = 0                  # gates "learned": the cell re-expressed through a random projection of each
                                   # port's state into this many dimensions (the user's projection; 0: off)
    copies: int = 1                # Net2Net-style widening: every compiled unit duplicated, outgoing weights / copies


def _param(shape, gen, dtype, std):
    return nn.Parameter(torch.randn(*shape, generator=gen, dtype=dtype) * std)


class Cell(nn.Module):
    """One kind's cell circuit as dense per-layer matrices with a mask (the allowed entries) and compiled values."""

    def __init__(self, circuit, S, cfg: HCfg, kd, learn: str, gen, dtype, table_simplex=False, B=3):
        super().__init__()
        units, outs = circuit
        self.S, self.B, self.kd = S, B, kd
        nz0 = 3 * S
        layers = sorted({u.layer for u in units})
        self.nlayers = len(layers)
        self.table_simplex = table_simplex
        copies = cfg.copies
        # unit u of the compiled circuit -> its copies' positions in z (copies side by side, Net2Net)
        sizes = [sum(1 for u in units if u.layer == l) for l in layers]
        widths = [s * copies for s in sizes]
        offs = np.cumsum([nz0] + widths)
        pos, count = {}, {l: 0 for l in layers}
        for i, u in enumerate(units):
            li = layers.index(u.layer)
            base = offs[li] + count[u.layer] * copies
            pos[i] = [base + c for c in range(copies)]
            count[u.layer] += 1
        self.layer_slices = [(int(offs[i]), int(offs[i + 1])) for i in range(len(layers))]
        self.outs = [pos[o][0] for o in outs]                         # read the first copy (all copies are equal)
        self.out_all = [pos[o] for o in outs]
        zmap = lambda k: list(range(k, k + 1)) if k < nz0 else pos[k - nz0]
        self.WL, self.WG, self.bL, self.bG, self.mL, self.mG, self.isg = [], [], [], [], [], [], []
        for li, l in enumerate(layers):
            zin = int(offs[li])
            width = widths[li]
            WL = torch.zeros(zin, width, dtype=dtype)
            WG = torch.zeros(zin, width, dtype=dtype)
            bL = torch.zeros(width, dtype=dtype)
            bG = torch.zeros(width, dtype=dtype)
            mL = torch.zeros(zin, width, dtype=torch.bool)
            mG = torch.zeros(zin, width, dtype=torch.bool)
            isg = torch.zeros(width, dtype=torch.bool)
            for i, u in enumerate(units):
                if u.layer != l:
                    continue
                for col in pos[i]:
                    c = col - zin
                    isg[c] = u.kind == "reglu"
                    bL[c] = u.lbias
                    bG[c] = u.gbias
                    for k, v in u.left.items():
                        srcs = zmap(k)
                        for s_ in srcs:                               # an input that has copies: split the weight
                            WL[s_, c] = v / len(srcs)
                            mL[s_, c] = True
                    for k, v in u.gate.items():
                        srcs = zmap(k)
                        for s_ in srcs:
                            WG[s_, c] = v / len(srcs)
                            mG[s_, c] = True
            self.isg.append(isg)
            self.mL.append(mL)
            self.mG.append(mG)
            self.WL.append(WL)
            self.WG.append(WG)
            self.bL.append(bL)
            self.bG.append(bG)
        self.learn = learn
        n = len(layers)
        if learn == "gates" and cfg.dense_out and n > 1:
            # the output rows read every unit of the last hidden layer: the table's pattern is learned too, not given
            # by the compiled sparsity (which joint product feeds which output)
            lo, hi = self.layer_slices[n - 2]
            self.mL[n - 1][lo:hi] = True
        for i in range(n):
            self.register_buffer(f"cWL{i}", self.WL[i])
            self.register_buffer(f"cWG{i}", self.WG[i])
            self.register_buffer(f"cbL{i}", self.bL[i])
            self.register_buffer(f"cbG{i}", self.bG[i])
            self.register_buffer(f"mL{i}", self.mL[i])
            self.register_buffer(f"mG{i}", self.mG[i])
            self.register_buffer(f"isg{i}", self.isg[i])
        self.p = nn.ParameterDict()
        self.lift = cfg.lift if learn == "gates" else 0
        if learn == "gates" and self.lift:
            # The user's projection: the cell re-expressed through a fixed random projection Q [d, S] of each port's
            # state (Gaussian, d > S). A weight row that reads port entries is the projection Q^T W' of a larger
            # trainable W' [3d, units] (the rows read the lifted ports Q p); the last layer writes a lifted state
            # y' = W'' z [d], read back through Q's pseudo-inverse. Exact at the compiled point (W' the minimal-norm
            # lift of the compiled weights), and dense where the compiled weights were sparse.
            d = self.lift
            gq = torch.Generator().manual_seed(999)
            Q = torch.randn(d, S, generator=gq, dtype=dtype) / math.sqrt(d)
            Qb = torch.block_diag(Q, Q, Q)                                  # [3d, 3S]: lifted ports = Qb p
            self.register_buffer("Qb", Qb)
            self.register_buffer("Qpinv", torch.linalg.pinv(Q))             # [S, d]
            Qb_left = torch.linalg.pinv(Qb.T)                               # W' = pinv(Qb^T) W: Qb^T W' = W
            for i in range(n):
                for name, W, std in (("WL", self.WL[i], cfg.init), ("WG", self.WG[i], cfg.init)):
                    port, rest = W[:nz0], W[nz0:]
                    if i == n - 1 and name == "WL":                        # the output layer writes lifted states
                        port, rest = port @ Q.T, rest @ Q.T
                    self.register_buffer(f"c{name}p{i}", Qb_left @ port)
                    self.register_buffer(f"c{name}r{i}", rest)
                    self.p[f"{name}p{i}"] = _param((3 * d, port.shape[1]), gen, dtype, std)
                    self.p[f"{name}r{i}"] = _param(rest.shape, gen, dtype, std)
                bl = self.bL[i] @ Q.T if i == n - 1 else self.bL[i]
                self.register_buffer(f"cbLl{i}", bl)
                self.p[f"bL{i}"] = _param(bl.shape, gen, dtype, cfg.init)
                self.p[f"bG{i}"] = _param(self.bG[i].shape, gen, dtype, cfg.init)
        elif learn == "gates":
            for i in range(n):
                std = cfg.init
                self.p[f"WL{i}"] = _param(self.WL[i].shape, gen, dtype, std)
                self.p[f"WG{i}"] = _param(self.WG[i].shape, gen, dtype, std)
                self.p[f"bL{i}"] = _param(self.bL[i].shape, gen, dtype, std)
                self.p[f"bG{i}"] = _param(self.bG[i].shape, gen, dtype, std)
        if learn == "table":                                          # only the last layer's rows over the joint units
            i = n - 1
            zin = self.WL[i].shape[0]
            if table_simplex:
                # joint units = the last hidden layer; rows: per joint unit a softmax over digits and over borrows
                lo, hi = self.layer_slices[n - 2]
                self.joint = (lo, hi)
                if cfg.tparam == "densetab":
                    # the rows are a dense ReGLU cell's logits on each joint unit's one-hot (a, b, c): the H1 form
                    # (the table is the generic cell's outputs on every one-hot input)
                    self.tabnet = DenseCell(S, [B * B, 2 * B * B], gen, dtype, cfg.init, "used")
                    self.register_buffer("joint_ports", self._joint_ports(lo, hi, dtype))
                else:
                    self.p["Ld"] = _param((hi - lo, B), gen, dtype, cfg.init)
                    self.p["Lb"] = _param((hi - lo, 2), gen, dtype, cfg.init)
            else:
                self.p["WL_t"] = _param(self.WL[i].shape, gen, dtype, cfg.init)
                self.p["bL_t"] = _param(self.bL[i].shape, gen, dtype, cfg.init)

    @torch.no_grad()
    def _joint_ports(self, lo, hi, dtype):
        """For each joint unit of the compiled circuit, the one-hot ports (a, b, c) on which it is 1."""
        B, S = self.B, self.S
        combos = [(a, b, c) for a in range(B) for b in range(B) for c in range(2)]
        P = torch.zeros(len(combos), 3 * S, dtype=dtype)
        for r, (a, b, c) in enumerate(combos):
            P[r, a] = 1
            P[r, S + b] = 1
            P[r, 2 * S + B + c] = 1
            P[r, S - 2] = P[r, 2 * S - 2] = 1                               # A's and B's borrow: none
            P[r, 2 * S] = 1                                                 # C's digit: 0
        z = P
        for i in range(self.nlayers - 1):
            WL, WG, bL, bG = (getattr(self, f"c{k}{i}") for k in ("WL", "WG", "bL", "bG"))
            v = z @ WL + bL
            isg = getattr(self, f"isg{i}")
            v = torch.where(isg, v * torch.relu(z @ WG + bG), v)
            z = torch.cat([z, v], -1)
        units = z[:, lo:hi]                                                 # [combos, joint units]
        assert bool(((units == 0) | (units == 1)).all()) and bool((units.sum(0) == 1).all())
        return P[units.argmax(0)]                                           # [joint units, 3 S]

    def weights(self, i, hard=False):
        """Layer i's (WL, WG, bL, bG) as used: compiled buffers, or learned values on the mask."""
        cWL, cWG = getattr(self, f"cWL{i}"), getattr(self, f"cWG{i}")
        cbL, cbG = getattr(self, f"cbL{i}"), getattr(self, f"cbG{i}")
        mL, mG = getattr(self, f"mL{i}"), getattr(self, f"mG{i}")
        if self.learn == "gates" and self.lift:
            # effective weights in the original basis: port rows Qb^T W' (dense), unit rows on the compiled mask; the
            # last layer's columns read back through Q's pseudo-inverse
            nz0 = 3 * self.S
            last = i == self.nlayers - 1
            out = []
            for name, m in (("WL", mL), ("WG", mG)):
                port = self.Qb.T @ self.p[f"{name}p{i}"]
                rest = self.p[f"{name}r{i}"]
                if last and name == "WL":
                    port, rest = port @ self.Qpinv.T, (rest @ self.Qpinv.T) * m[nz0:]
                else:
                    rest = rest * m[nz0:]
                out.append(torch.cat([port, rest], 0))
            bL = self.p[f"bL{i}"] @ self.Qpinv.T if last else self.p[f"bL{i}"]
            return out[0], out[1] * mG.any(0, keepdim=True), bL, self.p[f"bG{i}"]
        if self.learn == "gates":
            return (self.p[f"WL{i}"] * mL, self.p[f"WG{i}"] * mG, self.p[f"bL{i}"], self.p[f"bG{i}"])
        if self.learn == "table" and i == self.nlayers - 1:
            if self.table_simplex:
                lo, hi = self.joint
                B = self.B
                if hasattr(self, "tabnet"):
                    o = self.tabnet(self.joint_ports)
                    d, b = o[:, :B], o[:, B:]
                else:
                    d, b = self.p["Ld"], self.p["Lb"]
                if hard:
                    d = F.one_hot(d.argmax(-1), B).to(d.dtype)
                    b = F.one_hot(b.argmax(-1), 2).to(b.dtype)
                else:
                    d, b = d.softmax(-1), b.softmax(-1)
                WL = torch.zeros_like(cWL)
                WL[lo:hi] = torch.cat([d, b], -1)
                return WL, cWG, cbL * 0, cbG
            return self.p["WL_t"] * mL, cWG, self.p["bL_t"], cbG
        return cWL, cWG, cbL, cbG

    def forward(self, ports, hard=False):
        """ports [..., 3 S] -> the next state [..., S] (raw values)."""
        z = ports
        for i in range(self.nlayers):
            WL, WG, bL, bG = self.weights(i, hard)
            v = z @ WL + bL
            isg = getattr(self, f"isg{i}")
            if bool(isg.any()):
                g = torch.relu(z @ WG + bG)
                v = torch.where(isg, v * g, v)
            z = torch.cat([z, v], -1)
        if len(self.out_all[0]) == 1:
            return z[..., self.outs]
        return torch.stack([z[..., cs].mean(-1) for cs in self.out_all], -1)    # copies averaged

    def l1(self):
        return 0.0


class DenseCell(nn.Module):
    """A kind's cell with its own wiring freed: a generic ReGLU network over the cell's ports, shaped like handwired's
    layers (every layer reads the ports and every earlier layer): two ReGLU layers of the compiled SUB circuit's
    widths (times `hidden_mult`), then an affine output layer. Every weight is learned from random, under L1.
    `inputs` "full" reads every entry of the three ports; "used" only the entries the compiled cells read, A's and B's
    digits and C's borrow (the attention bridge's MLP cell's input)."""

    def __init__(self, S, widths, gen, dtype, std, inputs="full"):
        super().__init__()
        self.S = S
        B = S - 2
        self.sel = None if inputs == "full" else torch.tensor(list(range(B)) + list(range(S, S + B))
                                                              + list(range(2 * S + B, 3 * S)))
        zin = 3 * S if self.sel is None else len(self.sel)
        self.WL, self.WG, self.bL, self.bG = (nn.ParameterList() for _ in range(4))
        for wd in widths:
            self.WL.append(_param((zin, wd), gen, dtype, std))
            self.WG.append(_param((zin, wd), gen, dtype, std))
            self.bL.append(_param((wd,), gen, dtype, std))
            self.bG.append(_param((wd,), gen, dtype, std))
            zin += wd
        self.WO = _param((zin, S), gen, dtype, std)
        self.bO = _param((S,), gen, dtype, std)

    def forward(self, ports, hard=False):
        z = ports if self.sel is None else ports[..., self.sel]
        for WL, WG, bL, bG in zip(self.WL, self.WG, self.bL, self.bG):
            z = torch.cat([z, (z @ WL + bL) * torch.relu(z @ WG + bG)], -1)
        return z @ self.WO + self.bO

    def l1(self):
        return sum(W.abs().sum() for W in list(self.WL) + list(self.WG)) + self.WO.abs().sum()


class Block(nn.Module):
    """One tick's parameters: the three kinds' cells and the routing."""

    def __init__(self, cfg: HCfg, circuits, B, gen, dtype):
        super().__init__()
        S = B + 2
        self.cells = nn.ModuleDict()
        sub_widths = [sum(1 for u in circuits[SUB][0] if u.layer == l) for l in (1, 2)]
        for kd in SCRATCH:
            name = G.KINDS[kd]
            free = name in cfg.free_kinds
            if cfg.gates == "dense" and free:
                self.cells[name] = DenseCell(S, [wd * cfg.hidden_mult for wd in sub_widths], gen, dtype, cfg.init,
                                             cfg.dense_inputs)
                continue
            if cfg.gates == "learned" and free:
                learn = "gates"
            elif cfg.tables == "learned" and kd == SUB:
                learn = "table"
            else:
                learn = "compiled"
            self.cells[name] = Cell(circuits[kd], S, cfg, kd, learn, gen, dtype,
                                    table_simplex=(cfg.tparam == "simplex"), B=B)
        # Multiplexer gates: softmax and sigmoid start at logits ~ 0; relu and linear start at 1 / options (a port's
        # weights then add up to about 1, as the softmax's do), every gate positive (a relu gate below 0 gets no
        # gradient), with relative noise `init`.
        relu_like = cfg.mux in ("relu", "relu_free", "raw", "softplus")
        C = cfg.copies                                      # Net2Net: parallel copies of every port's gates
        if cfg.route == "menu":
            nopt = len(BN.menu_options(cfg))
            noise = torch.randn(3, C, NK, nopt, generator=gen, dtype=dtype) * cfg.init
            if relu_like:                                   # raw: weights ~ 1 / options; normalized: gates ~ 1
                g0 = cfg.gate_init or (1.0 / nopt if cfg.mux == "raw" else 1.0)
                self.menu_g = nn.Parameter(g0 * (1 + noise))
            else:
                self.menu_g = nn.Parameter(cfg.gate_init + noise)
        elif cfg.route == "dense":
            Rb, Cb = cfg.window
            self.nbr, self.nbc = Rb + 1, 2 * Cb + 1
            noise = torch.randn(3, C, NK, NK, self.nbr, self.nbc, 2, generator=gen, dtype=dtype) * cfg.init
            if relu_like:
                g0 = cfg.gate_init or (0.05 if cfg.mux == "raw" else 1.0)
                self.menu_dense = nn.Parameter(g0 * (1 + noise))
            else:
                self.menu_dense = nn.Parameter(cfg.gate_init + noise)


class RIndex:
    """Per-layout tensors for the routing (cached)."""

    def __init__(self, lay: G.Layout, cfg: HCfg):
        L = lay.L
        self.kind = torch.from_numpy(lay.kind)
        self.inp = torch.from_numpy(lay.is_input)
        self.comp = torch.from_numpy(np.where(lay.comp >= 0, lay.comp, 0))      # [3, L]; unused ports read cell 0
        if cfg.route == "menu":
            opts = BN.menu_options(cfg)
            src = np.zeros((len(opts), L), dtype=np.int64)
            for m, opt in enumerate(opts):
                for p in np.where(~lay.is_input)[0]:
                    src[m, p] = BN.menu_source(lay, p, opt)
            self.menu_src = torch.from_numpy(src)                                # [M, L]
        if cfg.route == "dense":
            Rb, Cb = cfg.window
            dr = lay.row[None, :] - lay.row[:, None]                            # key row - query row
            dc = lay.col[None, :] - lay.col[:, None]
            ok = dr <= 0
            if cfg.causal:
                order = lay.row * lay.C + lay.col
                ok &= order[None, :] < order[:, None]
            ok &= np.abs(dc) <= Cb + 10                                         # every column (clipped bins)
            self.ok = torch.from_numpy(ok & ~lay.is_input[:, None])
            self.br = torch.from_numpy(np.clip(-dr, 0, Rb))                     # 0 .. Rb (Rb = that far or further)
            self.bc = torch.from_numpy(np.clip(dc, -Cb, Cb) + Cb)
            self.top = torch.from_numpy(lay.top.astype(np.int64))[None, :].expand(L, L)
            self.qk = self.kind[:, None].expand(L, L)
            self.kk = self.kind[None, :].expand(L, L)


def gate_size(g, mux):
    return torch.sigmoid(g) if mux in ("sigmoid_free", "softmax") else torch.relu(g)


def groupnorm(x, eps=1e-9):
    x = x.clamp_min(0) + eps
    return x / x.sum(-1, keepdim=True)


class HNet(nn.Module):
    def __init__(self, cfg: HCfg, B: int, seed: int = 0, dtype=torch.float64, rec=None):
        super().__init__()
        self.cfg, self.B, self.dtype = cfg, B, dtype
        rec = rec or load.record(B, 2, 1)                                   # any size of this base: the same circuits
        self.circuits = extract(rec)
        g = torch.Generator().manual_seed(seed)
        nblocks = 1 if cfg.tied else cfg.nblocks
        assert cfg.tied or nblocks, "untied blocks need nblocks"
        self.blocks = nn.ModuleList([Block(cfg, self.circuits, B, g, dtype) for _ in range(nblocks)])
        self._idx = {}
        self.force = None
        self.rule_block = 0                                                 # the block kind_rule queries

    def index(self, lay):
        key = (lay.B, lay.w, lay.k, lay.links)
        if key not in self._idx:
            self._idx[key] = RIndex(lay, self.cfg)
        return self._idx[key]

    def block(self, t):
        return self.blocks[0] if self.cfg.tied else self.blocks[min(t, len(self.blocks) - 1)]

    # ----- the routing -----
    def mux(self, g, dim, hard, mask=None):
        """A port's weights over its sources from the gates g (`mask`: the sources it may read):
            softmax       exp(g) / sum: the attention bridge's competition (normalized, gradients centered and
                          weighted by each option's own share: the strong get stronger)
            relu          relu(g) / sum: handwired's multiplexer, value * relu(gate), normalized (gradients
                          centered, but not weighted by the option's share)
            relu_free     relu(g) / sum with the sum held constant in the backward pass: the forward pass reads a
                          distribution, and each gate's gradient is its own source's, independent of the others (no
                          competition at all)
            sigmoid_free  sigmoid(g), normalized the same way
            softplus      softplus(4 g) / sum: a ReLU multiplexer whose gates never reach 0 (a gate below 0 still gets a
                          gradient), normalized
            raw           relu(g), unnormalized (the plain multiplexer; with product cells over many ticks it explodes
                          unless the host normalizes what it writes back)
        """
        c = self.cfg
        if mask is not None:
            g = g.masked_fill(~mask, -math.inf if c.mux == "softmax" else -1e9)
        if hard:
            w = F.one_hot(g.argmax(dim), g.shape[dim]).to(g.dtype).movedim(-1, dim)
            return w if mask is None else w * mask.any(dim, keepdim=True)
        if c.mux == "softmax":
            w = g.softmax(dim)
            return w if mask is None else w.nan_to_num(0.0)
        r = (torch.sigmoid(g) if c.mux == "sigmoid_free" else F.softplus(g, beta=4.0) if c.mux == "softplus"
             else torch.relu(g))
        if mask is not None:
            r = r * mask
        if c.mux == "raw":
            return r
        tot = r.sum(dim, keepdim=True) + 1e-12
        if c.mux in ("relu_free", "sigmoid_free"):
            tot = tot.detach()
        return r / tot

    def routing(self, lay, ix, blk, hard):
        """[3, L, L] port-reading weights (rows: the reading cell)."""
        c = self.cfg
        L = lay.L
        if c.route == "fixed":
            return None
        if c.route == "menu":
            g = blk.menu_g                                                     # [3, copies, NK, M]
            if g.shape[1] == 1:
                w = self.mux(g[:, 0], -1, hard)                                # [3, NK, M]
            else:                                                              # a port reads its copies' mean
                w = self.mux(g, -1, False).mean(1)
                if hard:
                    w = F.one_hot(w.argmax(-1), w.shape[-1]).to(w.dtype)
            if self.force is not None:
                (h, kd), o = self.force
                w = w.clone()
                w[h, kd] = F.one_hot(torch.tensor(o), w.shape[-1]).to(w.dtype)
            wq = w[:, ix.kind]                                                 # [3, L, M]
            R = torch.zeros(3, L, L, dtype=self.dtype)
            idx = ix.menu_src.T[None].expand(3, L, -1)                         # [3, L, M]
            R = R.scatter_add(2, idx, wq)
            return R
        # dense: joint table over (query kind, key kind, row bin, column bin, top)
        D = blk.menu_dense
        if D.shape[1] == 1:
            T = D[:, 0, ix.qk, ix.kk, ix.br, ix.bc, ix.top]                     # [3, L, L]
            return self.mux(T, -1, hard, mask=ix.ok[None].expand_as(T))
        T = D[:, :, ix.qk, ix.kk, ix.br, ix.bc, ix.top]                         # [3, copies, L, L]
        w = self.mux(T, -1, False, mask=ix.ok[None, None].expand_as(T)).mean(1)
        if hard:
            w = w.masked_fill(~ix.ok[None], -1.0)
            w = F.one_hot(w.argmax(-1), L).to(w.dtype) * ix.ok.any(-1)[None, :, None]
        return w

    def ports(self, lay, ix, blk, S, hard):
        """[X, L, 3 S] the three ports of every cell."""
        R = self.routing(lay, ix, blk, hard)
        if R is None:
            P = S[:, ix.comp]                                                  # [X, 3, L, S]
        else:
            P = torch.einsum("hqk,xks->xhqs", R, S)
        return P.permute(0, 2, 1, 3).reshape(S.shape[0], lay.L, -1)

    # ----- a tick -----
    def tick(self, lay, ix, blk, S, hard):
        P = self.ports(lay, ix, blk, S, hard)
        out = S.clone()
        for kd in SCRATCH:
            m = ix.kind == kd
            if bool(m.any()):
                out[:, m] = blk.cells[G.KINDS[kd]](P[:, m], hard)
        return out

    def snap(self, S):
        return G.snap_state(S, self.B)

    def soft(self, S):
        """The soft write-back: each group of a state (digit, borrow) as a distribution."""
        B = self.B
        if self.cfg.writeback == "softmax":
            b = self.cfg.wb_beta
            return torch.cat([(b * S[..., :B]).softmax(-1), (b * S[..., B:]).softmax(-1)], -1)
        return torch.cat([groupnorm(S[..., :B]), groupnorm(S[..., B:])], -1)

    def inputs(self, lay, Np, vp):
        return G.input_states(lay, Np, vp, self.dtype)

    def forward(self, lay, S_in, hard=False, record=False, ticks=None):
        ix = self.index(lay)
        S = S_in
        hist = []
        for t in range(ticks or lay.depth):
            S = self.tick(lay, ix, self.block(t), S, hard)
            S = torch.where(ix.inp[None, :, None], S_in, S)
            if hard:
                S = self.snap(S)
            elif self.cfg.writeback != "raw":
                S = self.soft(S)
            if record:
                hist.append(S)
        return (S, hist) if record else S

    def answer_raw(self, lay, Np, vp, hard=False):
        S = self.forward(lay, self.inputs(lay, Np, vp), hard)
        return S[:, lay.answer, :self.B]

    # ----- the SUB cell on constructed ports (the cell lessons) -----
    def kind_rule(self, kd, a, b, c, hard=False):
        """Kind kd's cell on port distributions a [n, B], b [n, B], borrow c [n, 2] -> raw digit [n, B], borrow [n, 2]
        (the other entries of the ports: a blank digit 0 or no borrow)."""
        B = self.B
        n = a.shape[0]
        blank_d = torch.zeros(n, B, dtype=self.dtype)
        blank_d[:, 0] = 1
        blank_b = torch.zeros(n, 2, dtype=self.dtype)
        blank_b[:, 0] = 1
        ports = torch.cat([a, blank_b, b, blank_b, blank_d, c], -1)
        o = self.blocks[self.rule_block].cells[G.KINDS[kd]](ports, hard)
        return o[:, :B], o[:, B:]

    def l1(self):
        tot = 0.0
        for blk in self.blocks:
            for cell in blk.cells.values():
                tot = tot + cell.l1()
            # sparsity pressure on the gates themselves (a normalized port's weights always sum to 1)
            if self.cfg.route == "dense":
                tot = tot + gate_size(blk.menu_dense, self.cfg.mux).sum()
            elif self.cfg.route == "menu":
                tot = tot + gate_size(blk.menu_g, self.cfg.mux).sum()
        return tot


def check_against_program(sizes=((3, 2, 1), (3, 2, 2), (4, 2, 2), (3, 3, 2), (5, 2, 2)), sample=4000):
    """The regrouped network at its compiled point against handwired's compiled layers (load.Program): the same
    final state at every cell, on every question (sampled above `sample`), and the same cell circuits at every size."""
    out = []
    for B, w, k in sizes:
        rec = load.record(B, w, k)
        same_circuits = all(same_circuit(a, b) for a, b in zip(extract(rec).values(), extract(load.record(B, 2, 1)).values()))
        net = HNet(HCfg(), B, rec=load.record(B, 2, 1))
        prog = load.Program(rec)
        lay = G.Layout(B, w, k)
        data = G.Data(B, w, k, sample=sample, seed=1)
        S0 = G.data_inputs(lay, data)
        with torch.no_grad():
            a = net(lay, S0)
            b = prog.run(S0)
        out.append({"size": [B, w, k], "questions": int(data.X), "same_circuits": bool(same_circuits),
                    "max_abs_diff": float((a - b).abs().max()),
                    "answers": int((a[:, lay.answer, :B].argmax(-1) == data.yd).all(-1).sum())})
    return out
