"""eng_calc, basic chemistry (D48): molar mass, stoichiometry, acids and bases, solutions, kinetics,
electrochemistry.

Atomic weights: CIAAW/IUPAC abridged table (`data/atomic_weights.json`, with the source and the hash
of the files it was extracted from). Constants R and F: CODATA 2022 (`constants.py`). Concentrations in
mol/L, also as a number without units (exception to the SI rule, D48). The chemical formula is read by a
character parser: no eval.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from decimal import Decimal
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

from ..constants import lookup
from ..envelope import Ctx
from ..errors import ToolError
from ..fmt import fmt_float
from .eng_base import (
    choice,
    integer,
    log_ratio,
    need,
    one_of,
    only,
    op,
    qty,
    temperature,
    units_for,
)

_DATA = Path(__file__).resolve().parents[1] / "data" / "atomic_weights.json"
R_GAS = lookup("R")[1]
FARADAY = lookup("F")[1]
KW_25 = 1.0e-14  # ionic product of water at 25 degrees C, handbook value (D48)
_MAX_FORMULA = 200
_MAX_DEPTH = 10


@cache
def _elements() -> dict[str, dict[str, Any]]:
    return json.loads(_DATA.read_text(encoding="utf-8"))["elements"]


def _num(s: str, i: int) -> tuple[int, int]:
    j = i
    while j < len(s) and s[j].isdigit():
        j += 1
    if j == i:
        return 1, i
    n = int(s[i:j])
    if not 0 < n <= 10**6:
        raise ToolError("INVALID_ARGUMENT", f"Index {n} outside 1-1,000,000 in the formula.")
    return n, j


def _group(s: str, i: int, depth: int) -> tuple[Counter[str], int]:
    counts: Counter[str] = Counter()
    els = _elements()
    while i < len(s):
        ch = s[i]
        if ch in "([":
            if depth >= _MAX_DEPTH:
                raise ToolError("INVALID_ARGUMENT", "Formula with too many nested brackets.")
            close = ")" if ch == "(" else "]"
            inner, i = _group(s, i + 1, depth + 1)
            if i >= len(s) or s[i] != close:
                raise ToolError("INVALID_ARGUMENT", f"Bracket '{ch}' not closed in the formula.")
            n, i = _num(s, i + 1)
            for k, v in inner.items():
                counts[k] += v * n
        elif ch in ")]":
            if depth == 0:
                raise ToolError("INVALID_ARGUMENT", f"Extra bracket '{ch}' in the formula.")
            return counts, i
        elif ch.isupper():
            j = i + 1
            while j < len(s) and s[j].islower():
                j += 1
            sym = s[i:j]
            if sym not in els:
                raise ToolError("INVALID_ARGUMENT", f"Unknown element '{sym}' in the formula.")
            n, i = _num(s, j)
            counts[sym] += n
        else:
            raise ToolError(
                "INVALID_ARGUMENT",
                f"Character '{ch}' not allowed in the formula: element symbols, numbers, () and [], "
                "· or * for hydrates.",
            )
    if depth > 0:
        raise ToolError("INVALID_ARGUMENT", "Bracket not closed in the formula.")
    return counts, i


def parse_formula(text: Any) -> dict[str, int]:
    """Counts the atoms of a formula: H2O, Ca(OH)2, K4[Fe(CN)6], CuSO4·5H2O (also CuSO4*5H2O)."""
    if not isinstance(text, str) or not 0 < len(text.strip()) <= _MAX_FORMULA:
        raise ToolError(
            "INVALID_ARGUMENT", f"The formula is a text of 1-{_MAX_FORMULA} characters.", '"H2SO4"'
        )
    total: Counter[str] = Counter()
    for part in re.split(r"[·•*.]", text.replace(" ", "")):
        if not part:
            raise ToolError("INVALID_ARGUMENT", "Formula with an empty part (hydrate separator).")
        m = re.match(r"\d+", part)
        coef = int(m.group()) if m else 1
        body = part[m.end() :] if m else part
        if not body or not 0 < coef <= 10**6:
            raise ToolError("INVALID_ARGUMENT", f"Invalid part of the formula: '{part}'.")
        counts, _ = _group(body, 0, 0)
        for k, v in counts.items():
            total[k] += coef * v
    return dict(total)


def molar_mass_of(counts: dict[str, int]) -> tuple[Decimal, Decimal]:
    """Molar mass in g/mol and uncertainty (linear sum of the uncertainties of the abridged weights)."""
    els = _elements()
    M = U = Decimal(0)
    for sym, n in counts.items():
        ab = els[sym]["abridged"]
        if not ab:
            raise ToolError(
                "INVALID_ARGUMENT",
                f"{sym} ({els[sym]['name']}) has no standard atomic weight: no stable isotope. "
                "Use the mass of a nuclide.",
            )
        M += n * Decimal(ab["value"])
        U += n * Decimal(ab["uncertainty"])
    return M, U


def _molar_mass_param(p: dict[str, Any]) -> float:
    """Molar mass in kg/mol from `formula` or from `M`."""
    key = one_of(p, ("formula", "M"), "the chemical formula or the molar mass M")
    if key == "M":
        return need(p, "M", "molar_mass")
    return float(molar_mass_of(parse_formula(p["formula"]))[0]) / 1000


# ------------------------------------------------------------------ molar_mass


@op("molar_mass", ("formula", "m", "n"))
def _molar_mass(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    counts = parse_formula(p.get("formula"))
    Mg, Ug = molar_mass_of(counts)
    M = float(Mg) / 1000
    out: dict[str, Any] = {
        "formula": p["formula"],
        "elements": counts,
        "M": M,
        "M_g_mol": float(Mg),
        "M_uncertainty_g_mol": float(Ug),
        "mass_fraction": {
            s: round(float(n * Decimal(_elements()[s]["abridged"]["value"]) / Mg), 12)
            for s, n in counts.items()
        },
    }
    formulas = ["M = sum(n_i A_r,i) g/mol, CIAAW/IUPAC abridged atomic weights"]
    if p.get("m") is not None and p.get("n") is not None:
        raise ToolError("INVALID_ARGUMENT", "Give m or n, not both.")
    if p.get("m") is not None:
        out["n"] = need(p, "m", "mass") / M
        formulas.append("n = m / M")
    elif p.get("n") is not None:
        out["m"] = need(p, "n", "amount_of_substance") * M
        formulas.append("m = n M")
    ctx.formula = "; ".join(formulas)
    ctx.formatted = f"Molar mass of {p['formula']}: {fmt_float(float(Mg), 4)} g/mol."
    ctx.units = units_for(
        out,
        {"M": "kg/mol", "M_g_mol": "g/mol", "M_uncertainty_g_mol": "g/mol", "n": "mol", "m": "kg"},
    )
    return out


# ------------------------------------------------------------------ reaction_stoich

_ARROWS = ("<=>", "⇌", "->", "→", "=")
_MOL_UNITS = {"mol": 1.0, "kmol": 1e3, "mmol": 1e-3}


def _split_reaction(text: Any) -> tuple[list[str], list[str]]:
    if not isinstance(text, str) or not 0 < len(text) <= 500:
        raise ToolError("INVALID_ARGUMENT", 'params.reaction: text like "H2 + O2 -> H2O".')
    arrow = next((a for a in _ARROWS if a in text), None)
    if arrow is None:
        raise ToolError(
            "INVALID_ARGUMENT",
            "The reaction arrow is missing (->, →, =, <=>).",
            '"H2 + O2 -> H2O"',
        )
    left, _, right = text.partition(arrow)

    def side(t: str) -> list[str]:
        out = []
        for s in t.split("+"):
            s = re.sub(r"^\s*\d+\s*", "", s).strip()
            if not s:
                raise ToolError("INVALID_ARGUMENT", "Empty species in the reaction: check the '+'.")
            out.append(s)
        return out

    return side(left), side(right)


@op("reaction_stoich", ("reaction", "amounts"))
def _stoich(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    reac, prod = _split_reaction(p.get("reaction"))
    species = reac + prod
    if len(set(species)) != len(species):
        raise ToolError("INVALID_ARGUMENT", "The same species appears twice in the reaction.")
    counts = [parse_formula(s) for s in species]
    elems = sorted({e for c in counts for e in c})
    A = sp.Matrix(
        [
            [(1 if j < len(reac) else -1) * counts[j].get(el, 0) for j in range(len(species))]
            for el in elems
        ]
    )
    ns = A.nullspace()
    if len(ns) != 1:
        raise ToolError(
            "INVALID_ARGUMENT",
            "The reaction does not balance in a unique way"
            + (
                ": no solution (check the formulas)."
                if not ns
                else f" ({len(ns)} independent balances)."
            ),
        )
    v = ns[0]
    den = sp.ilcm(*[sp.fraction(sp.nsimplify(x))[1] for x in v])
    ints = [int(x * den) for x in v]
    if all(k < 0 for k in ints):
        ints = [-k for k in ints]
    if any(k <= 0 for k in ints):
        raise ToolError(
            "INVALID_ARGUMENT", "The reaction does not balance with all positive coefficients."
        )
    g = math.gcd(*ints)
    coef = [k // g for k in ints]
    Ms = [float(molar_mass_of(c)[0]) / 1000 for c in counts]

    def term(k: int, s: str) -> str:
        return s if k == 1 else f"{k} {s}"

    balanced = (
        " + ".join(term(k, s) for k, s in zip(coef[: len(reac)], reac, strict=True))
        + " -> "
        + " + ".join(term(k, s) for k, s in zip(coef[len(reac) :], prod, strict=True))
    )
    out: dict[str, Any] = {
        "reactants": reac,
        "products": prod,
        "species": species,
        "coefficients": coef,
        "balanced": balanced,
        "M_g_mol": [m * 1000 for m in Ms],
    }
    formulas = [
        "coefficients: smallest integer vector of the null space of the element x species matrix"
    ]
    amounts = p.get("amounts")
    if amounts is not None:
        if not isinstance(amounts, dict) or not amounts:
            raise ToolError(
                "INVALID_ARGUMENT",
                'params.amounts: object {"species": amount}.',
                '{"O2": [32, "g"]}',
            )
        given: dict[int, float] = {}
        for s, raw in amounts.items():
            if s not in reac:
                raise ToolError(
                    "INVALID_ARGUMENT",
                    f"params.amounts: '{s}' is not a reactant ({', '.join(reac)}).",
                )
            i = species.index(s)
            if isinstance(raw, list | tuple) and len(raw) == 2 and raw[1] in _MOL_UNITS:
                val = qty({"x": raw[0]}, "x", None)
                assert val is not None
                given[i] = val * _MOL_UNITS[str(raw[1])]
            elif isinstance(raw, list | tuple):
                given[i] = need({s: raw}, s, "mass") / Ms[i]
            else:
                given[i] = need({s: raw}, s, None)
        lim = min(given, key=lambda i: given[i] / coef[i])
        xi = given[lim] / coef[lim]
        n_r = [k * xi for k in coef]
        out |= {
            "limiting": species[lim],
            "extent": xi,
            "n_reacted": n_r,
            "mass_reacted": [n * m for n, m in zip(n_r, Ms, strict=True)],
            "n_left": [given[i] - n_r[i] if i in given else None for i in range(len(species))],
        }
        formulas.append("extent xi = min(n_i / nu_i) over the given reactants; n_j = nu_j xi")
    ctx.formula = "; ".join(formulas)
    ctx.formatted = f"Balanced reaction: {balanced}." + (
        f" Limiting reactant: {out['limiting']}." if "limiting" in out else ""
    )
    ctx.units = units_for(
        out,
        {
            "M_g_mol": "g/mol",
            "extent": "mol",
            "n_reacted": "mol",
            "mass_reacted": "kg",
            "n_left": "mol",
        },
    )
    return out


# ------------------------------------------------------------------ acid_base


def _positive_root(coeffs: list[float]) -> float:
    """Unique positive root of the polynomial (signs + + - -), refined with Newton."""
    roots = [
        r.real for r in np.roots(coeffs) if abs(r.imag) <= 1e-12 * max(1.0, abs(r)) and r.real > 0
    ]
    if len(roots) != 1:
        raise ToolError("MATH_ERROR", "Charge balance without a unique positive solution.")
    h = roots[0]
    poly, dpoly = np.poly1d(coeffs), np.poly1d(coeffs).deriv()
    for _ in range(6):
        d = float(dpoly(h))
        if d == 0:
            break
        h -= float(poly(h)) / d
    return h


def _conc(p: dict[str, Any], key: str) -> float:
    return need(p, key, "concentration")


def _k_from(p: dict[str, Any], k: str, pk: str) -> float:
    key = one_of(p, (k, pk), f"{k} or {pk}")
    return need(p, k, None) if key == k else 10 ** -need(p, pk, None, positive=False)


_AB = {
    "strong": {"kind", "c", "Kw"},
    "weak": {"kind", "c", "Ka", "pKa", "Kb", "pKb", "Kw"},
    "buffer": {"c_acid", "c_base", "Ka", "pKa", "Kw"},
}


@op("acid_base", ("mode", "kind", "c", "Ka", "pKa", "Kb", "pKb", "c_acid", "c_base", "Kw"))
def _acid_base(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    mode = choice(p, "mode", tuple(_AB))
    only(p, mode, _AB[mode])
    Kw = qty(p, "Kw", None, default=KW_25) or KW_25
    out: dict[str, Any] = {}
    if mode == "buffer":
        Ka = _k_from(p, "Ka", "pKa")
        ca, cb = _conc(p, "c_acid"), _conc(p, "c_base")
        h = _positive_root([1.0, Ka + cb, -(Ka * ca + Kw), -Kw * Ka])
        out["pH_henderson"] = -math.log10(Ka) + math.log10(cb / ca)
        formulas = [
            "charge balance [H+] + c_base = Ka (c_acid + c_base) / (Ka + [H+]) + Kw / [H+] (exact)",
            "Henderson-Hasselbalch pH = pKa + log10(c_base / c_acid) (approximate)",
        ]
    else:
        kind = choice(p, "kind", ("acid", "base"))
        c = _conc(p, "c")
        if mode == "strong":
            x = (c + math.sqrt(c * c + 4 * Kw)) / 2
            approx = -math.log10(c)
            formulas = [
                "[H+] (or [OH-]) = (c + sqrt(c^2 + 4 Kw)) / 2, with the self-ionization of water"
            ]
        else:
            if kind == "acid" and (p.get("Kb") is not None or p.get("pKb") is not None):
                raise ToolError("INVALID_ARGUMENT", "For a weak acid give Ka or pKa.")
            if kind == "base" and (p.get("Ka") is not None or p.get("pKa") is not None):
                raise ToolError("INVALID_ARGUMENT", "For a weak base give Kb or pKb.")
            K = _k_from(p, "Ka", "pKa") if kind == "acid" else _k_from(p, "Kb", "pKb")
            x = _positive_root([1.0, K, -(K * c + Kw), -K * Kw])
            out["alpha"] = K / (K + x)
            approx = 0.5 * (-math.log10(K) - math.log10(c))
            formulas = [
                "x^3 + K x^2 - (K c + Kw) x - K Kw = 0 (exact charge balance, x = [H+] or [OH-])",
                "approximate: p = (pK - log10 c) / 2",
            ]
        h = x if kind == "acid" else Kw / x
        pKw = -math.log10(Kw)
        out["pH_approx"] = approx if kind == "acid" else pKw - approx
    pH = -math.log10(h)
    out = {"pH": pH, "pOH": -math.log10(Kw / h), "H": h, "OH": Kw / h} | out
    ctx.formula = "; ".join(formulas) + f"; Kw = {Kw:g}"
    ctx.formatted = f"pH = {fmt_float(pH, 3)}."
    if p.get("Kw") is None:
        ctx.warn("Kw = 1.0e-14: valid at 25 degrees C; at another temperature pass params.Kw.")
    ctx.units = units_for(
        out,
        {
            "pH": "1",
            "pOH": "1",
            "H": "mol/L",
            "OH": "mol/L",
            "alpha": "1",
            "pH_approx": "1",
            "pH_henderson": "1",
        },
    )
    return out


# ------------------------------------------------------------------ solution_prep

_SOL = {"dilution": {"c1", "V1", "c2", "V2"}, "molarity": {"formula", "M", "m", "n", "c", "V"}}


@op("solution_prep", ("mode", "c1", "V1", "c2", "V2", "formula", "M", "m", "n", "c", "V"))
def _solution(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    mode = choice(p, "mode", tuple(_SOL))
    only(p, mode, _SOL[mode])
    out: dict[str, Any]
    if mode == "dilution":
        missing = [k for k in ("c1", "V1", "c2", "V2") if p.get(k) is None]
        if len(missing) != 1:
            raise ToolError(
                "INVALID_ARGUMENT", "Give three of c1, V1, c2, V2: the fourth is the result."
            )
        v = {
            k: (_conc(p, k) if k.startswith("c") else need(p, k, "volume"))
            for k in ("c1", "V1", "c2", "V2")
            if k not in missing
        }
        m = missing[0]
        if m == "c1":
            v[m] = v["c2"] * v["V2"] / v["V1"]
        elif m == "V1":
            v[m] = v["c2"] * v["V2"] / v["c1"]
        elif m == "c2":
            v[m] = v["c1"] * v["V1"] / v["V2"]
        else:
            v[m] = v["c1"] * v["V1"] / v["c2"]
        out = {m: v[m], "n": v["c1"] * v["V1"] * 1000}
        ctx.formula = "c1 V1 = c2 V2; n = c1 V1"
        unit = "mL" if m.startswith("V") else "mol/L"
        ctx.formatted = f"{m} = {fmt_float(v[m] * (1e6 if m.startswith('V') else 1), 6)} {unit}."
    else:
        V = need(p, "V", "volume")
        given = [k for k in ("m", "n", "c") if p.get(k) is not None]
        if len(given) != 1:
            raise ToolError(
                "INVALID_ARGUMENT", "With V give one of m (mass), n (moles), c (concentration)."
            )
        needs_M = given[0] == "m" or p.get("formula") is not None or p.get("M") is not None
        M = _molar_mass_param(p) if needs_M else None
        if given[0] == "m":
            assert M is not None
            n = need(p, "m", "mass") / M
        elif given[0] == "n":
            n = need(p, "n", "amount_of_substance")
        else:
            n = _conc(p, "c") * V * 1000
        out = {"c": n / (V * 1000), "n": n}
        if M is not None:
            out |= {"m": n * M, "M": M}
        ctx.formula = "c = n / V (V in L); n = m / M"
        ctx.formatted = f"Concentration {fmt_float(out['c'], 6)} mol/L, {fmt_float(n, 6)} mol."
    ctx.units = units_for(
        out,
        {
            "c1": "mol/L",
            "c2": "mol/L",
            "c": "mol/L",
            "V1": "m3",
            "V2": "m3",
            "n": "mol",
            "m": "kg",
            "M": "kg/mol",
        },
    )
    return out


# ------------------------------------------------------------------ reaction_kinetics

_KIN = {
    "arrhenius": {"A", "Ea", "T", "k1", "T1", "k2", "T2"},
    "integrated": {"order", "k", "c0", "t", "c"},
}


@op(
    "reaction_kinetics",
    ("mode", "A", "Ea", "T", "k1", "T1", "k2", "T2", "order", "k", "c0", "t", "c"),
)
def _kinetics(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    mode = choice(p, "mode", tuple(_KIN))
    only(p, mode, _KIN[mode])
    out: dict[str, Any]
    if mode == "arrhenius":
        have = {k for k in _KIN["arrhenius"] if p.get(k) is not None}
        if have == {"A", "Ea", "T"}:
            T = temperature(p, "T")
            assert T is not None
            out = {"k": need(p, "A", None) * math.exp(-need(p, "Ea", "molar_energy") / (R_GAS * T))}
            ctx.formula = "k = A exp(-Ea / (R T))"
        elif have == {"k1", "T1", "k2", "T2"}:
            T1, T2 = temperature(p, "T1"), temperature(p, "T2")
            assert T1 is not None and T2 is not None
            k1, k2 = need(p, "k1", None), need(p, "k2", None)
            if T1 == T2:
                raise ToolError("INVALID_ARGUMENT", "T1 and T2 must be different.")
            # 1/T1 - 1/T2 = (T2 - T1) / (T1 T2) and ln(k2 / k1) with log1p: no cancellation (D79)
            Ea = R_GAS * log_ratio(k2, k1) * T1 * T2 / (T2 - T1)
            out = {"Ea": Ea, "A": k1 * math.exp(Ea / (R_GAS * T1))}
            ctx.formula = "Ea = R ln(k2 / k1) / (1/T1 - 1/T2); A = k1 exp(Ea / (R T1))"
        elif have == {"k1", "T1", "Ea", "T2"}:
            T1, T2 = temperature(p, "T1"), temperature(p, "T2")
            assert T1 is not None and T2 is not None
            k1, Ea = need(p, "k1", None), need(p, "Ea", "molar_energy")
            out = {
                "k2": k1 * math.exp(-Ea / R_GAS * (1 / T2 - 1 / T1)),
                "A": k1 * math.exp(Ea / (R_GAS * T1)),
            }
            ctx.formula = "k2 = k1 exp(-Ea/R (1/T2 - 1/T1))"
        else:
            raise ToolError(
                "INVALID_ARGUMENT",
                "Arrhenius: give A, Ea, T or k1, T1, k2, T2 or k1, T1, Ea, T2.",
            )
        ctx.formatted = ", ".join(f"{k} = {v:.6g}" for k, v in out.items()) + "."
        ctx.units = units_for(out, {"Ea": "J/mol"})
        return out
    order = integer(p, "order", 0)
    if order > 2:
        raise ToolError("INVALID_ARGUMENT", "params.order: 0, 1 or 2.")
    k, c0 = need(p, "k", None), _conc(p, "c0")
    half = {0: c0 / (2 * k), 1: math.log(2) / k, 2: 1 / (k * c0)}[order]
    if (p.get("t") is None) == (p.get("c") is None):
        raise ToolError("INVALID_ARGUMENT", "Give t (time) or c (concentration to reach).")
    if p.get("t") is not None:
        t = need(p, "t", "time", positive=False)
        if order == 0:
            c = max(c0 - k * t, 0.0)
            if c == 0.0:
                ctx.warn("Zero order: the reactant is used up before t; c = 0.")
        elif order == 1:
            c = c0 * math.exp(-k * t)
        else:
            c = 1 / (1 / c0 + k * t)
        out = {"c": c, "t": t}
    else:
        c = _conc(p, "c")
        if not c < c0:
            raise ToolError("INVALID_ARGUMENT", "params.c must be smaller than c0.")
        # c close to c0: c0 - c is exact there, so no form subtracts two rounded numbers (D77)
        if order == 0:
            t = (c0 - c) / k
        elif order == 1:
            t = math.log1p((c0 - c) / c) / k  # = ln(c0 / c); (c0 - c) / c >= 0, never close to -1
        else:
            t = (c0 - c) / (c * c0) / k
        out = {"t": t, "c": c}
    out["t_half"] = half
    ctx.formula = {
        0: "c = c0 - k t; t_1/2 = c0 / (2 k)",
        1: "c = c0 exp(-k t); t_1/2 = ln 2 / k",
        2: "1/c = 1/c0 + k t; t_1/2 = 1 / (k c0)",
    }[order]
    ctx.formatted = (
        f"c = {fmt_float(out['c'], 6)} mol/L after {fmt_float(out['t'], 4)} s; "
        f"half-life {fmt_float(half, 4)} s."
    )
    ctx.units = {"c": "mol/L", "t": "s", "t_half": "s"}
    return out


# ------------------------------------------------------------------ electrochemistry

_EC = {
    "nernst": {"E0", "n", "Q", "T"},
    "electrolysis": {"I", "t", "charge", "n", "formula", "M"},
    "gibbs": {"n", "E", "dG", "K", "T"},
}


@op(
    "electrochemistry",
    ("mode", "E0", "n", "Q", "T", "I", "t", "charge", "formula", "M", "E", "dG", "K"),
)
def _electrochem(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    mode = choice(p, "mode", tuple(_EC))
    only(p, mode, _EC[mode])
    n = integer(p, "n", 1)
    out: dict[str, Any]
    if mode == "electrolysis":
        if p.get("charge") is not None:
            if p.get("I") is not None or p.get("t") is not None:
                raise ToolError("INVALID_ARGUMENT", "Give charge or I with t.")
            Qc = need(p, "charge", "charge")
        else:
            Qc = need(p, "I", "current") * need(p, "t", "time")
        M = _molar_mass_param(p)
        mol = Qc / (n * FARADAY)
        out = {"charge": Qc, "n_mol": mol, "m": mol * M}
        ctx.formula = "n = Q / (z F); m = n M (Faraday's law)"
        ctx.formatted = f"Deposited mass {fmt_float(mol * M * 1000, 4)} g."
        ctx.units = {"charge": "C", "n_mol": "mol", "m": "kg"}
        return out
    T = temperature(p, "T", need=False) or 298.15
    if mode == "nernst":
        E0, Q = need(p, "E0", "voltage", positive=False), need(p, "Q", None)
        E = E0 - R_GAS * T / (n * FARADAY) * math.log(Q)
        out = {"E": E, "slope_per_decade": R_GAS * T * math.log(10) / (n * FARADAY), "T": T}
        ctx.formula = "E = E0 - (R T / (n F)) ln Q"
        ctx.formatted = f"E = {fmt_float(E, 5)} V."
        ctx.units = {"E": "V", "slope_per_decade": "V", "T": "K"}
        return out
    key = one_of(p, ("E", "dG", "K"), "E, dG or K")
    if key == "E":
        E = need(p, "E", "voltage", positive=False)
        dG = -n * FARADAY * E
    elif key == "dG":
        dG = need(p, "dG", "molar_energy", positive=False)
        E = -dG / (n * FARADAY)
    else:
        dG = -R_GAS * T * math.log(need(p, "K", None))
        E = -dG / (n * FARADAY)
    lnK = -dG / (R_GAS * T)
    if lnK > 700:
        raise ToolError(
            "MATH_ERROR", f"K = exp({lnK:.4g}) exceeds the largest floating-point number."
        )
    out = {"E": E, "dG": dG, "K": math.exp(lnK), "lnK": lnK, "T": T}
    ctx.formula = "dG = -n F E; dG = -R T ln K"
    ctx.formatted = (
        f"dG = {fmt_float(dG / 1000, 4)} kJ/mol, E = {fmt_float(E, 5)} V, K = {math.exp(lnK):.6g}."
    )
    ctx.units = {"E": "V", "dG": "J/mol", "K": "1", "lnK": "1", "T": "K"}
    return out
