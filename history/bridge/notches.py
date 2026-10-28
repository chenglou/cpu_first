"""The notches, from the compiled machine to a plain looped transformer (net.Cfg fields), and the lesson tiers.

Each notch loosens one thing relative to the one above it.
"""

NOTCHES = {
    # tables learned, everything else compiled
    "N1_table": dict(attn="fixed", rule="table", learn=("SUB",)),            # the machine's learned table
    "N1_tables": dict(attn="fixed", rule="table", learn=("SHIFT", "SUB", "SEL")),   # every rule a learned table
    "N1_mlp": dict(attn="fixed", rule="mlp"),                                # one ReLU MLP for every kind
    "N1_mlp_wide": dict(attn="fixed", rule="mlp", hidden=256),
    "N1_mlp_st": dict(attn="fixed", rule="mlp", st=True),                    # + straight-through snapped states
    "N1_mlp_sub": dict(attn="fixed", rule="mlp", mlp_kinds=("SUB",)),       # only the subtraction cell an MLP
    "N1_mlp_sub_st": dict(attn="fixed", rule="mlp", mlp_kinds=("SUB",), st=True),
    "N1_mlptab_sub": dict(attn="fixed", rule="mlptab", learn=("SUB",)),       # the SUB table is an MLP's outputs
    "N1_mlp_sub_w256": dict(attn="fixed", rule="mlp", mlp_kinds=("SUB",), hidden=256),
    "N1_mlptab_sub_w256": dict(attn="fixed", rule="mlptab", learn=("SUB",), hidden=256),
    "N2_menu_mlp_sub": dict(attn="menu", rule="mlp", mlp_kinds=("SUB",)),
    "N2_menu_mlptab_sub": dict(attn="menu", rule="mlptab", learn=("SUB",), hidden=256),
    # the attention loosened
    "N2_menu": dict(attn="menu", rule="table", learn=("SUB",)),              # each head picks from 7 relative rules
    "N2_menu_given": dict(attn="menu", rule="table", learn=()),              # the truth tables given: wiring only
    "N2_rel_given": dict(attn="menu", menu="rel", rule="table", learn=()),   # + every relative offset (~23 rules)
    "N2_rel": dict(attn="menu", menu="rel", rule="table", learn=("SUB",)),
    "N2_rel13_given": dict(attn="menu", menu="rel", menu_extra=6, rule="table", learn=()),   # 13 options per port
    "N2_narrow": dict(attn="menu", menu="narrow", rule="table", learn=("SUB",)),   # TINYROUTING-like switches
    "N2_narrow_given": dict(attn="menu", menu="narrow", rule="table", learn=()),
    "N2_ncarry_given": dict(attn="menu", menu="narrow_carry", rule="table", learn=()),
    "N2_wnocarry_given": dict(attn="menu", menu="wide_nocarry", rule="table", learn=()),
    "N3_free_given": dict(attn="free", rule="table", learn=()),
    "N3_free_given_st": dict(attn="free", rule="table", learn=(), st=True),     # + snapped states (straight-through)
    "N3_free_given_first": dict(attn="free", rule="table", learn=(), inject="first"),   # the prompt at tick 0 only
    "N3_free_given_first_st": dict(attn="free", rule="table", learn=(), inject="first", st=True),
    "N3_free_first": dict(attn="free", rule="table", learn=("SUB",), inject="first"),
    "N3_rows_given": dict(attn="free", rule="table", learn=(), window=(2, 99)),   # 2 rows up, any column + N row
    "N3_rows": dict(attn="free", rule="table", learn=("SUB",), window=(2, 99)),
    "N2_menu_mlp": dict(attn="menu", rule="mlp"),
    "N3_free": dict(attn="free", rule="table", learn=("SUB",)),              # each head attends anywhere (soft)
    "N3_free_mlp": dict(attn="free", rule="mlp"),
    # the cell's format loosened: states are symbols with learned meanings, heads have value projections
    "N4_symbols": dict(attn="free", rule="mlp", fmt="symbols", K=6),          # K = 2B symbols (one per digit/borrow)
    "N4_symbols_untied": dict(attn="free", rule="mlp", fmt="symbols", tied=False),
    "N4_symbols_wide": dict(attn="free", rule="mlp", fmt="symbols", K=10, hidden=256),
    # continuous states (a standard residual block), kinds kept
    "N5_free": dict(attn="free", rule="mlp", fmt="free", d=16, hidden=128),
    # a plain looped transformer: learned absolute positions, content attention, no scratch kinds
    "N6_plain": dict(attn="abs", rule="mlp", fmt="free", roles="none", d=32, hidden=128),
}

# the widened twins of the notches with a compiled point (the Net2Net arm, suites wstab / wlearn): head copies,
# Net2Net-widened MLPs (the `widen` of a compiled start), a lifted symbol alphabet
WIDE = {
    "N1_mlp": (dict(hidden=216), 4),
    "N2_menu_given": (dict(copies=4), 1),
    "N2_rel13_given": (dict(copies=4), 1),
    "N3_free_given": (dict(copies=4), 1),
    "N3_free_given_first": (dict(copies=4), 1),
    "N3_free_mlp": (dict(copies=4, copy_mode="concat", hidden=216), 4),
    "N4_symbols": (dict(K=24, copies=4, copy_mode="concat", hidden=216), 4),
}

# lessons a notch can express; cell-level lessons need an identifiable subtraction cell (fmt digits, roles kinds)
TIERS = {
    "A": {"ans": 1.0},                                                       # the answers alone
    "AC": {"ans": 1.0, "pref": 1.0, "colall": 1.0, "rowall": 1.0, "bij": 1.0},                          # + the cell's own order facts
    "G": {"ans": 1.0, "xmodn": 1.0, "step": 1.0},                            # + network-level laws
    "GC": {"ans": 1.0, "xmodn": 1.0, "step": 1.0, "pref": 1.0, "colall": 1.0, "rowall": 1.0, "bij": 1.0},
    "AS": {"ans": 1.0, "step": 1.0},                                         # + the step law only
    "CD": {"ans": 1.0, "cellD": 1.0},                                        # TINYROUTING's reference recipe
    "C": {"pref": 1.0, "colall": 1.0, "rowall": 1.0, "bij": 1.0},
    "ACnb": {"ans": 1.0, "pref": 1.0, "colall": 1.0, "rowall": 1.0},                # AC without bijectivity                                       # the cell's order facts alone
    "AL": {"ans": 1.0, "link": 1.0},                                         # + one link's own laws (a sub-grid)
    "ACL": {"ans": 1.0, "pref": 1.0, "colall": 1.0, "rowall": 1.0, "bij": 1.0, "link": 1.0},
    "GCL": {"ans": 1.0, "xmodn": 1.0, "step": 1.0, "pref": 1.0, "colall": 1.0, "rowall": 1.0, "bij": 1.0, "link": 1.0},
    "GL": {"ans": 1.0, "xmodn": 1.0, "step": 1.0, "link": 1.0},
}


def cell_level_ok(net_cfg: dict) -> bool:
    return net_cfg.get("fmt", "digits") == "digits" and net_cfg.get("roles", "kinds") == "kinds"
