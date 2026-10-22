"""The handwired bridge's notches (hnet.HCfg fields) and lesson tiers (the attention bridge's, `bridge.notches.TIERS`).

Each notch frees one more part of the compiled network and trains it from random; the rest stays compiled.
"""

from bridge.notches import TIERS  # noqa: F401  (the same lessons as the attention bridge)

# How a freed port weighs its sources (hnet.HNet.mux). "raw" is the plain multiplexer, value * relu(gate), without any
# normalization; it needs the host to normalize what it writes back after each tick (raw product cells explode).
MUXES = {
    "softmax": dict(mux="softmax"),
    "relu": dict(mux="relu"),
    "relu_free": dict(mux="relu_free"),
    "sigmoid_free": dict(mux="sigmoid_free"),
    "raw": dict(mux="raw", writeback="norm"),
    "softplus": dict(mux="softplus"),
}
SOFT = dict(writeback="softmax")      # generic cell weights: the host writes back softmax(10 * values) per group

NOTCHES = {
    "H0_compiled": dict(),
    # the tables: the subtraction cell's output rows over its 18 joint products
    "H1_table": dict(tables="learned", tparam="simplex"),          # each joint product's row a softmax (the bridge's)
    "H1_table_raw": dict(tables="learned", tparam="raw", **SOFT),  # plain affine weights
    # the gates and multiplexers: every unit's values in the cell's circuit, the cell's wiring kept
    "H2_gates_sub": dict(gates="learned", free_kinds=("SUB",), **SOFT),
    "H2_gates_sub_i1": dict(gates="learned", free_kinds=("SUB",), init=1.0, **SOFT),   # values ~ N(0, 1)
    "H2_gates": dict(gates="learned", **SOFT),
    # the same circuit with its output rows dense over the 18 joint products: which product feeds which output (the
    # table's pattern) is learned too; minimal and in 4 Net2Net copies
    "H2_gates_sub_do": dict(gates="learned", free_kinds=("SUB",), dense_out=True, **SOFT),
    "H2_gates_sub_do_x4": dict(gates="learned", free_kinds=("SUB",), dense_out=True, copies=4, **SOFT),
    "H2_gates_sub_x4": dict(gates="learned", free_kinds=("SUB",), copies=4, **SOFT),
    # the user's projection: the same circuit re-expressed through a random projection of each port's state into 20 or
    # 40 dimensions (4 or 8 times the state), its port-reading weights dense in the lifted basis
    "H2_gates_sub_lift20": dict(gates="learned", free_kinds=("SUB",), lift=20, **SOFT),
    "H2_gates_sub_lift40": dict(gates="learned", free_kinds=("SUB",), lift=40, **SOFT),
    # the cell's own wiring freed too: a generic ReGLU network over the port entries the compiled cells read (A's and
    # B's digits, C's borrow: the attention bridge's MLP cell), or over every entry of the three ports ("full")
    "H2_dense_sub": dict(gates="dense", free_kinds=("SUB",), dense_inputs="used", **SOFT),
    "H2_dense_sub_full": dict(gates="dense", free_kinds=("SUB",), dense_inputs="full", **SOFT),
    "H2_dense": dict(gates="dense", dense_inputs="used", **SOFT),
    # the dense subtraction cell with a softer write-back (softmax of 3 x values) and 4 times wider
    "H2_dense_sub_b3": dict(gates="dense", free_kinds=("SUB",), dense_inputs="used", writeback="softmax", wb_beta=3.0),
    "H2_dense_sub_x4": dict(gates="dense", free_kinds=("SUB",), dense_inputs="used", hidden_mult=4, **SOFT),
    # the H1 form: the subtraction table is a dense ReGLU cell's outputs on every one-hot input (the compiled products
    # read that table)
    "H2_densetab_sub": dict(tables="learned", tparam="densetab"),
    # the routing: which cells each port reads
    **{f"H3_menu_given_{m}": dict(route="menu", **v) for m, v in MUXES.items()},
    **{f"H3_menu_{m}": dict(route="menu", tables="learned", **v) for m, v in MUXES.items()},
    **{f"H3_rel{n}_given_{m}": dict(route="menu", menu="rel", menu_extra=n - 7, **v) for m, v in MUXES.items()
       for n in (10, 13, 16, 20)},
    **{f"H3_dense_given_{m}": dict(route="dense", **v) for m, v in MUXES.items()},
    **{f"H3_dense_{m}": dict(route="dense", tables="learned", **v) for m, v in MUXES.items()},
    # the loop untied: one block per tick
    "H4_untied_table": dict(tables="learned", tied=False),
    "H4_untied_dense_sub": dict(gates="dense", free_kinds=("SUB",), dense_inputs="used", tied=False, **SOFT),
    "H4_untied_menu_given_softmax": dict(route="menu", mux="softmax", tied=False),
    "H4_untied_menu_given_relu": dict(route="menu", mux="relu", tied=False),
}


def needs_low_lr(net: dict) -> bool:
    """Generic cell weights (gates, dense cells) train at lr 0.01 like the attention bridge's MLP cells; tables and
    routing at 0.05 like its tables and menus."""
    return net.get("gates", "compiled") != "compiled" or net.get("tparam") == "densetab"


def readout(net: dict) -> str:
    """How the lessons read a digit group: as the write-back does."""
    return "softmax" if net.get("writeback") == "softmax" else "relu"
