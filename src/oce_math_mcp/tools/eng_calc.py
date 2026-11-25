"""eng_calc: engineering formulas (spec V3 section 12, D39).

The formula lives in the server: the model picks the operation and passes the data; the server
applies the formula, returns SI results with `units` and the formula used in `formula_used`.

The operations live in modules per area (eng_struct: area A; eng_mech: area B; eng_thermo: area C;
eng_elec: area D; eng_hydro: area E; eng_gas: area F; eng_tables: operations with tables, D44;
eng_solid: strength without tables, D46; eng_sci: constants, D47; eng_chem: chemistry, D48;
eng_nuc: nuclear physics, eng_modern: modern physics and relativity, D49; eng_steam: water and steam
IAPWS-IF97, D50; eng_psychro: psychrometrics, D51; eng_conv: Nusselt correlations, D52;
eng_hx: e-NTU heat exchangers, fins, transient, D53; eng_rad: radiation between surfaces, D54;
eng_em: electrostatics and capacitance, D55;
eng_mag: magnetism, inductance and induction, D56;
eng_linkage: mechanisms and epicyclic gear trains, D57;
eng_cable: catenary and plain bearings, D58;
eng_truss: trusses, D59;
eng_torsion: torsion of non-circular sections, D60;
eng_optics: geometric and wave optics, D64;
eng_electronics: practical electronics, D65;
eng_friction: friction in simple machines, D66;
eng_machines: electric machines, D67;
eng_plates: thin plates, D68;
eng_beams: continuous beams, D69;
eng_losses: local losses, Borda-Carnot, D70;
eng_holes: stress concentration at holes, D71;
eng_balance: unbalance and rotor balancing, D72;
eng_rolling: rolling without slipping, D73); this module loads them and
exposes `handle` and `OPS`.
"""

from __future__ import annotations

from . import (
    eng_balance,
    eng_beams,
    eng_cable,
    eng_chem,
    eng_conv,
    eng_elec,
    eng_electronics,
    eng_em,
    eng_friction,
    eng_gas,
    eng_holes,
    eng_hx,
    eng_hydro,
    eng_linkage,
    eng_losses,
    eng_machines,
    eng_mag,
    eng_mech,
    eng_modern,
    eng_nuc,
    eng_optics,
    eng_plates,
    eng_psychro,
    eng_rad,
    eng_rolling,
    eng_sci,
    eng_solid,
    eng_steam,
    eng_struct,
    eng_tables,
    eng_thermo,
    eng_torsion,
    eng_truss,
)
from .eng_base import OPS, handle

__all__ = [
    "OPS",
    "eng_balance",
    "eng_beams",
    "eng_cable",
    "eng_chem",
    "eng_conv",
    "eng_elec",
    "eng_electronics",
    "eng_em",
    "eng_friction",
    "eng_gas",
    "eng_holes",
    "eng_hx",
    "eng_hydro",
    "eng_linkage",
    "eng_losses",
    "eng_machines",
    "eng_mag",
    "eng_mech",
    "eng_modern",
    "eng_nuc",
    "eng_optics",
    "eng_plates",
    "eng_psychro",
    "eng_rad",
    "eng_rolling",
    "eng_sci",
    "eng_solid",
    "eng_steam",
    "eng_struct",
    "eng_tables",
    "eng_thermo",
    "eng_torsion",
    "eng_truss",
    "handle",
]
