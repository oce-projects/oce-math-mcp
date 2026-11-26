"""eng_calc: electronegativity, Pauling and Mulliken (D110).

Source: Wikipedia "Defining equation (physical chemistry)", section "Quantum chemistry", and the article
"Electronegativity", in the local offline library. Energies in eV per bond or per atom.
- Pauling: |chi_A - chi_B| = (eV)^(-1/2) sqrt(E_d(AB) - [E_d(AA) + E_d(BB)] / 2); the article's more accurate fit
  uses the geometric mean, E_d(AB) = sqrt(E_d(AA) E_d(BB)) + 1.3 (chi_A - chi_B)^2 eV. The square root gives the
  difference only: A is taken as the more electronegative atom, so chi_A = chi_B + delta_chi.
- The inverse (article): E_d(AB) = mean + (chi_A - chi_B)^2 eV, or with the geometric mean and the factor 1.3.
- Mulliken: absolute electronegativity (E_I + E_EA) / 2, chemical potential -chi; on the Pauling scale
  chi = 10^-3 [187 (E_I + E_EA) + 170] with the energies in eV (the article's 0.187 (E_i + E_ea) + 0.17).
Energies may be given per particle (J, eV, ...) or per mole (kJ/mol, kcal/mol: divided by N_A). A plain number
is in joules, as everywhere in eng_calc; a value far from chemical energies (outside 1e-3 to 1e3 eV) is flagged.
"""

from __future__ import annotations

import math
from typing import Any

from ..constants import lookup
from ..envelope import Ctx
from ..errors import ToolError
from .eng_base import choice, fmt_sig, need, only, op, qty, range_guard, units_for

_MODES: dict[str, set[str]] = {
    "pauling": {"E_AB", "E_AA", "E_BB", "mean", "chi_B"},
    "bond_energy": {"delta_chi", "chi_A", "chi_B", "E_AA", "E_BB", "mean"},
    "mulliken": {"E_I", "E_EA"},
}
_MEANS = ("arithmetic", "geometric")
_GEOMETRIC_FACTOR = 1.3  # eV per unit of (chi_A - chi_B)^2 in the geometric-mean fit (article)
_UNITS = {
    "mean_energy_eV": "eV",
    "excess_energy_eV": "eV",
    "delta_chi": "1",
    "chi_A": "1",
    "E_AB_eV": "eV",
    "E_AB_kJ_mol": "kJ/mol",
    "E_sum_eV": "eV",
    "chi_absolute_eV": "eV",
    "mu_chemical_eV": "eV",
    "chi_pauling_scale": "1",
}


def _ev() -> float:
    return lookup("e")[1]


def _energy_ev(p: dict[str, Any], key: str, ctx: Ctx, *, positive: bool = True) -> float:
    """An energy in eV, from a per-particle (J, eV, ...) or a per-mole (kJ/mol, ...) value."""
    raw = p.get(key)
    if (
        isinstance(raw, list)
        and len(raw) == 2
        and isinstance(raw[1], str)
        and raw[1].endswith("/mol")
    ):
        joule = need(p, key, "molar_energy", positive=positive) / lookup("N_A")[1]
    else:
        joule = need(p, key, "energy", positive=positive)
    value = joule / _ev()
    if value != 0 and not 1e-3 <= abs(value) <= 1e3:
        ctx.warn(
            f"params.{key} = {fmt_sig(value)} eV is far from chemical energies (1e-3 to 1e3 eV): "
            'a plain number is in joules; give [value, "eV"] or [value, "kJ/mol"].'
        )
    return value


def _mean(p: dict[str, Any], ctx: Ctx) -> tuple[str, float]:
    kind = choice(p, "mean", _MEANS, default="arithmetic")
    e_aa, e_bb = _energy_ev(p, "E_AA", ctx), _energy_ev(p, "E_BB", ctx)
    return kind, (e_aa + e_bb) / 2 if kind == "arithmetic" else math.sqrt(e_aa * e_bb)


@op("electronegativity", ("mode", *sorted(set().union(*_MODES.values()))))
@range_guard
def _electronegativity(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    mode = choice(p, "mode", tuple(_MODES))
    only(p, mode, _MODES[mode])
    out = {"pauling": _pauling, "bond_energy": _bond_energy, "mulliken": _mulliken}[mode](p, ctx)
    ctx.units = units_for(out, _UNITS)
    return out


def _pauling(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    e_ab = _energy_ev(p, "E_AB", ctx)
    kind, mean = _mean(p, ctx)
    excess = e_ab - mean
    if excess < 0:
        raise ToolError(
            "INVALID_ARGUMENT",
            f"E_AB = {fmt_sig(e_ab)} eV is below the mean of E_AA and E_BB ({fmt_sig(mean)} eV): no ionic "
            'excess energy, so no Pauling difference (the geometric mean, params.mean = "geometric", is often lower).',
        )
    delta = math.sqrt(excess if kind == "arithmetic" else excess / _GEOMETRIC_FACTOR)
    out: dict[str, Any] = {"mean_energy_eV": mean, "excess_energy_eV": excess, "delta_chi": delta}
    chi_b = qty(p, "chi_B", None, need=False, positive=False)
    if chi_b is not None:
        out["chi_A"] = chi_b + delta
    ctx.formula = (
        "|chi_A - chi_B| = (eV)^(-1/2) sqrt(E_d(AB) - [E_d(AA) + E_d(BB)] / 2)"
        if kind == "arithmetic"
        else "|chi_A - chi_B| = sqrt((E_d(AB) - sqrt(E_d(AA) E_d(BB))) / (1.3 eV))"
    ) + "; A is the more electronegative atom: chi_A = chi_B + delta_chi"
    ctx.formatted = f"delta chi = {fmt_sig(delta, 4)}"
    return out


def _bond_energy(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    if p.get("delta_chi") is not None:
        if p.get("chi_A") is not None or p.get("chi_B") is not None:
            raise ToolError(
                "INVALID_ARGUMENT",
                "Give params.delta_chi or params.chi_A and params.chi_B, not both.",
            )
        delta = need(p, "delta_chi", None, positive=False)
        if delta < 0:
            raise ToolError(
                "INVALID_ARGUMENT", "params.delta_chi is |chi_A - chi_B|, so it is >= 0."
            )
    else:
        if p.get("chi_A") is None or p.get("chi_B") is None:
            raise ToolError(
                "INVALID_ARGUMENT", "Give params.delta_chi, or params.chi_A and params.chi_B."
            )
        delta = abs(need(p, "chi_A", None, positive=False) - need(p, "chi_B", None, positive=False))
    kind, mean = _mean(p, ctx)
    factor = 1.0 if kind == "arithmetic" else _GEOMETRIC_FACTOR
    e_ab = mean + factor * delta * delta
    ctx.formula = (
        "E_d(AB) = [E_d(AA) + E_d(BB)] / 2 + (chi_A - chi_B)^2 eV"
        if kind == "arithmetic"
        else "E_d(AB) = sqrt(E_d(AA) E_d(BB)) + 1.3 (chi_A - chi_B)^2 eV"
    )
    ctx.formatted = f"E_d(AB) = {fmt_sig(e_ab)} eV"
    return {
        "delta_chi": delta,
        "mean_energy_eV": mean,
        "E_AB_eV": e_ab,
        "E_AB_kJ_mol": e_ab * _ev() * lookup("N_A")[1] / 1000,
    }


def _mulliken(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    total = _energy_ev(p, "E_I", ctx) + _energy_ev(p, "E_EA", ctx, positive=False)
    chi_abs = total / 2
    ctx.formula = (
        "chi = (E_I + E_EA) / 2 (absolute), mu = -chi; Pauling scale chi = 10^-3 [187 (E_I + E_EA) + 170], "
        "energies in eV"
    )
    ctx.formatted = f"chi = {fmt_sig(chi_abs)} eV absolute"
    return {
        "E_sum_eV": total,
        "chi_absolute_eV": chi_abs,
        "mu_chemical_eV": -chi_abs,
        "chi_pauling_scale": 1e-3 * (187 * total + 170),
    }
