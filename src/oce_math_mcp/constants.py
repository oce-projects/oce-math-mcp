"""CODATA physical constants (D47).

Values, units and uncertainties of `scipy.constants.physical_constants` (edition recorded in
`CODATA`, today CODATA 2022, the same as NIST). The values become exact rationals from their
decimal writing: the constants that define the SI since 2019 stay exact in computations too.
"""

from __future__ import annotations

import difflib

import scipy.constants as _sc
import sympy as sp
from scipy.constants import _codata

from .errors import ToolError

CODATA: str = str(_codata._current_codata)  # pyright: ignore[reportPrivateUsage]
SOURCE = "https://physics.nist.gov/cuu/Constants/"
TABLE: dict[str, tuple[float, str, float]] = {
    k: (float(v[0]), str(v[1]), float(v[2])) for k, v in _sc.physical_constants.items()
}
_LOWER = {k.lower(): k for k in TABLE}

# short symbol -> CODATA name; every entry is checked by the tests
ALIASES: dict[str, str] = {
    "c": "speed of light in vacuum",
    "h": "Planck constant",
    "hbar": "reduced Planck constant",
    "e": "elementary charge",
    "N_A": "Avogadro constant",
    "k_B": "Boltzmann constant",
    "R": "molar gas constant",
    "F": "Faraday constant",
    "G": "Newtonian constant of gravitation",
    "g_n": "standard acceleration of gravity",
    "m_e": "electron mass",
    "m_p": "proton mass",
    "m_n": "neutron mass",
    "m_u": "atomic mass constant",
    "u": "unified atomic mass unit",
    "m_d": "deuteron mass",
    "m_alpha": "alpha particle mass",
    "alpha": "fine-structure constant",
    "R_inf": "Rydberg constant",
    "a_0": "Bohr radius",
    "r_e": "classical electron radius",
    "lambda_C": "Compton wavelength",
    "E_h": "Hartree energy",
    "epsilon_0": "vacuum electric permittivity",
    "mu_0": "vacuum mag. permeability",
    "Z_0": "characteristic impedance of vacuum",
    "sigma": "Stefan-Boltzmann constant",
    "b_wien": "Wien wavelength displacement law constant",
    "c1": "first radiation constant",
    "c2": "second radiation constant",
    "mu_B": "Bohr magneton",
    "mu_N": "nuclear magneton",
    "Phi_0": "mag. flux quantum",
    "K_J": "Josephson constant",
    "R_K": "von Klitzing constant",
    "atm": "standard atmosphere",
    "eV": "electron volt",
    "V_m": "molar volume of ideal gas (273.15 K, 101.325 kPa)",
    "m_e_MeV": "electron mass energy equivalent in MeV",
    "m_p_MeV": "proton mass energy equivalent in MeV",
    "m_n_MeV": "neutron mass energy equivalent in MeV",
    "u_MeV": "atomic mass constant energy equivalent in MeV",
}
SYMBOL = {v: k for k, v in ALIASES.items()}
# Other names of a short symbol (D108): c_0 is the symbol of the speed of light in vacuum on the page
# "List of relativistic equations". Kept apart from ALIASES, so SYMBOL still gives "c".
OTHER_NAMES: dict[str, str] = {"c_0": "c", "c0": "c"}


def lookup(name: str) -> tuple[str, float, str, float]:
    """CODATA name, value, unit, standard uncertainty; accepts short symbols and any letter case."""
    if not isinstance(name, str) or not name.strip():
        raise ToolError("INVALID_ARGUMENT", "The name of the constant is a text.", "const('h')")
    short = OTHER_NAMES.get(name.strip(), name.strip())
    key = ALIASES.get(short) or _LOWER.get(name.strip().lower())
    if key is None:
        near = difflib.get_close_matches(name.strip().lower(), list(_LOWER), n=5, cutoff=0.6)
        hint = ", ".join(_LOWER[x] for x in near) or ", ".join(list(ALIASES)[:12])
        raise ToolError(
            "INVALID_ARGUMENT",
            f"Unknown constant '{name}'. Perhaps: {hint}. Short symbols: {', '.join(ALIASES)}.",
            "const('h')",
        )
    v, unit, unc = TABLE[key]
    return key, v, unit, unc


def exact_value(name: str) -> sp.Expr:
    """Value as the exact rational of the CODATA decimal writing (for math_eval)."""
    _, v, _, _ = lookup(name)
    return sp.Rational(repr(v))


def search(text: str, limit: int = 25) -> list[str]:
    t = text.strip().lower()
    if not t:
        raise ToolError("INVALID_ARGUMENT", "params.search: text to search for in the names.")
    return [k for k in TABLE if t in k.lower()][:limit]
