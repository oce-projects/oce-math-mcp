"""eng_calc, nuclear physics (D49): nuclide data AME2020 and NUBASE2020, radioactive decay,
nuclear reactions and Q value.

Data in `data/nuclides.json.gz` (extracted from the files of the evaluation, CC BY 3.0 licence of the
papers): atomic mass excesses in keV (AME2020 for the ground states, NUBASE2020 for the
isomers), half-lives, spin and parity, decay modes, isotopic abundances.

Q and thresholds use the nuclear masses `M_atom - Z m_e` and count electrons and
positrons explicitly; the binding energy of the electrons is neglected, as in the Q tables of AME2020.
The years of NUBASE2020 are tropical years (365.2422 d); the years written in the parameters are
Julian (365.25 d, eng_base).

Liquid-drop model (D115): nuclear radius r = r0 A^(1/3) with r0 = 1.2 fm and the semi-empirical binding energy
E_B = a_V A - a_S A^(2/3) - a_C Z (Z - 1) / A^(1/3) - a_A (N - Z)^2 / A + delta a_P A^k_P (delta = +1 for even N
and Z, -1 for odd N and Z, 0 for odd A), both from "List of equations in nuclear and particle physics"; the
coefficients are the column "Least-squares fit (1)" of "Semi-empirical mass formula" (15.8, 18.3, 0.714, 23.2,
12 MeV, k_P = -1/2), the set of the list page: over the 2,455 measured ground states with A >= 20 it is off from
AME2020 by 3.855 MeV rms (the other columns with Z (Z - 1) by 7.2-8.6 MeV; the table's footnotes do not say which
columns use Z^2 or (Z - A/2)^2).
"""

from __future__ import annotations

import gzip
import json
import math
import re
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..constants import lookup
from ..envelope import Ctx
from ..errors import ToolError
from .eng_base import choice, fmt_sig, integer, need, one_of, only, op, qty, units_for

_DATA = Path(__file__).resolve().parents[1] / "data" / "nuclides.json.gz"
ME_KEV = lookup("electron mass energy equivalent in MeV")[1] * 1e3
U_KEV = lookup("atomic mass constant energy equivalent in MeV")[1] * 1e3
U_KG = lookup("atomic mass constant")[1]
M_U = lookup("molar mass constant")[1]
N_A = lookup("Avogadro constant")[1]
KEV_J = 1.602176634e-16
YEAR_TROPICAL = 365.2422 * 86400.0
_Y = YEAR_TROPICAL
_T_UNIT = {
    "ys": 1e-24,
    "zs": 1e-21,
    "as": 1e-18,
    "fs": 1e-15,
    "ps": 1e-12,
    "ns": 1e-9,
    "us": 1e-6,
    "ms": 1e-3,
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
    "d": 86400.0,
    "y": _Y,
    "ky": 1e3 * _Y,
    "My": 1e6 * _Y,
    "Gy": 1e9 * _Y,
    "Ty": 1e12 * _Y,
    "Py": 1e15 * _Y,
    "Ey": 1e18 * _Y,
    "Zy": 1e21 * _Y,
    "Yy": 1e24 * _Y,
}
# indices of the record fields (see "fields" in the data file)
I_Z, I_A, I_ME, I_DME, I_EST, I_AM, I_DAM, I_BA, I_DBA, I_EXC, I_T, I_TU, I_DT, I_JPI, I_BR = range(
    15
)
_EXAMPLE = '{"nuclide": "U-235"}'


@lru_cache(maxsize=1)
def _table() -> dict[str, Any]:
    with gzip.open(_DATA, "rt", encoding="utf-8") as f:
        return json.load(f)


def _nuclides() -> dict[str, list[Any]]:
    return _table()["nuclides"]


@lru_cache(maxsize=1)
def _symbols() -> dict[str, str]:
    """lowercase symbol -> symbol of the table ("co" -> "Co")."""
    return {k.split("-")[0].lower(): k.split("-")[0] for k in _nuclides()}


# ------------------------------------------------------------------ nuclide names

_ALIAS = {
    "n": "n-1",
    "p": "H-1",
    "d": "H-2",
    "D": "H-2",
    "t": "H-3",
    "T": "H-3",
    "α": "He-4",
}
_WORDS = {
    "neutron": "n-1",
    "proton": "H-1",
    "deuteron": "H-2",
    "triton": "H-3",
    "alpha": "He-4",
}
_RX_SYM_A = re.compile(r"^([A-Za-z]{1,2})[- ]?(\d{1,3})([mn]?)$")
_RX_A_SYM = re.compile(r"^(\d{1,3})([mn]?)[- ]?([A-Za-z]{1,2})$")


def _try_nuclide(text: str) -> str | None:
    t = text.strip()
    if t in _ALIAS:
        return _ALIAS[t]
    if t.lower() in _WORDS:
        return _WORDS[t.lower()]
    m = _RX_SYM_A.match(t)
    if m:
        sym, a, iso = m.groups()
    else:
        m = _RX_A_SYM.match(t)
        if not m:
            return None
        a, iso, sym = m.groups()
    canon = _symbols().get(sym.lower())
    if canon is None:
        return None
    key = f"{canon}-{int(a)}{iso}"
    return key if key in _nuclides() else None


def nuclide_key(text: Any) -> str:
    """Name of the nuclide in the table: "U-235", "235U", "Tc-99m", "alpha", "p", "n"."""
    if not isinstance(text, str) or not 0 < len(text.strip()) <= 20:
        raise ToolError(
            "INVALID_ARGUMENT",
            'The nuclide is a text like "U-235", "235U", "Tc-99m", "alpha" or "n".',
            _EXAMPLE,
        )
    key = _try_nuclide(text)
    if key is not None:
        return key
    m = _RX_SYM_A.match(text.strip()) or _RX_A_SYM.match(text.strip())
    sym = None
    if m:
        g = m.groups()
        sym = _symbols().get((g[0] if g[0][0].isalpha() else g[2]).lower())
    if sym is not None:
        masses = sorted(
            int(k.split("-")[1].rstrip("mn")) for k in _nuclides() if k.startswith(sym + "-")
        )
        raise ToolError(
            "INVALID_ARGUMENT",
            f"'{text}' is not in NUBASE2020: for {sym} the table goes from A = {masses[0]} to A = {masses[-1]} "
            "(isomers with m or n, like Tc-99m).",
        )
    raise ToolError(
        "INVALID_ARGUMENT",
        f'Nuclide not recognised: \'{text}\'. Write symbol and mass number, like "U-235" or "235U".',
        _EXAMPLE,
    )


def _rec(key: str) -> list[Any]:
    return _nuclides()[key]


def nuclide_za(key: str) -> tuple[int, int]:
    r = _rec(key)
    return r[I_Z], r[I_A]


def half_life(rec: list[Any]) -> tuple[float | None, str, bool]:
    """(half-life in s, text of the source, approximate value); inf = stable."""
    raw, unit, unc = rec[I_T], rec[I_TU], rec[I_DT]
    if raw == "stbl":
        return math.inf, "stable", False
    if raw == "p-unst":
        return None, "unstable by particle emission (time not measured)", True
    if not raw:
        return None, "not known", True
    approx = any(c in raw for c in "<>~#") or not unc
    try:
        v = float(re.sub(r"[<>~#]", "", raw))
    except ValueError:
        return None, f"{raw} {unit}".strip(), True
    if unit not in _T_UNIT:
        return None, f"{raw} {unit}".strip(), True
    text = f"{raw} {unit}" + (f" ± {unc}" if unc[:1].isdigit() else (f" ({unc})" if unc else ""))
    return v * _T_UNIT[unit], text, approx


def mass_excess(key: str) -> tuple[float, float, bool]:
    r = _rec(key)
    return float(r[I_ME]), float(r[I_DME] or 0), bool(r[I_EST])


def nuclear_mass_kev(key: str) -> float:
    """Nuclear mass in keV (A u + excess - Z m_e)."""
    r = _rec(key)
    return r[I_A] * U_KEV + float(r[I_ME]) - r[I_Z] * ME_KEV


def molar_mass_kg(key: str) -> float:
    r = _rec(key)
    am = r[I_AM]
    u = float(Decimal(am) / Decimal(10**6)) if am else r[I_A] + float(r[I_ME]) / U_KEV
    return u * M_U


# ------------------------------------------------------------------ nuclide_data

_SEMF_PAGE = {"a_V": 15.8, "a_S": 18.3, "a_C": 0.714, "a_A": 23.2, "a_P": 12.0, "k_P": -0.5}
_R0 = 1.2e-15  # m
_SEMF_LIGHT = 20  # below this A the liquid drop fits poorly (no shell structure)
_MODEL_UNITS = {
    "r0_fm": "fm",
    "radius_fm": "fm",
    "semf_volume_MeV": "MeV",
    "semf_surface_MeV": "MeV",
    "semf_coulomb_MeV": "MeV",
    "semf_asymmetry_MeV": "MeV",
    "semf_pairing_MeV": "MeV",
    "semf_binding_MeV": "MeV",
    "semf_binding_per_nucleon_MeV": "MeV",
    "semf_minus_ame_MeV": "MeV",
    "semf_error_percent": "%",
}


@lru_cache(maxsize=1)
def _ground_by_za() -> dict[tuple[int, int], str]:
    return {(r[I_Z], r[I_A]): k for k, r in _nuclides().items() if not r[I_EXC]}


def _semf_coefficients(p: dict[str, Any]) -> dict[str, float]:
    c = dict(_SEMF_PAGE)
    raw = p.get("semf")
    if raw is None:
        return c
    if not isinstance(raw, dict) or not raw or any(k not in c for k in raw):
        raise ToolError(
            "INVALID_ARGUMENT",
            f"params.semf: an object with some of {', '.join(c)} (MeV; k_P is the exponent of A in the "
            "pairing term); the others keep the page values.",
            '{"semf": {"a_P": 34, "k_P": -0.75}}',
        )
    for k, v in raw.items():
        if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v):
            raise ToolError("INVALID_ARGUMENT", f"params.semf.{k} must be a finite number.")
        c[k] = float(v)
    return c


def _liquid_drop(z: int, n: int, p: dict[str, Any], ctx: Ctx) -> tuple[dict[str, Any], str]:
    """D115: radius r0 A^(1/3) and the semi-empirical binding energy, term by term."""
    a = z + n
    r0 = qty(p, "r0", "length", default=_R0)
    assert r0 is not None
    c = _semf_coefficients(p)
    if z % 2 == 0 and n % 2 == 0:
        delta = 1
    elif z % 2 == 1 and n % 2 == 1:
        delta = -1
    else:
        delta = 0
    terms = {
        "semf_volume_MeV": c["a_V"] * a,
        "semf_surface_MeV": -c["a_S"] * a ** (2 / 3),
        "semf_coulomb_MeV": -c["a_C"] * z * (z - 1) / a ** (1 / 3),
        "semf_asymmetry_MeV": -c["a_A"] * (n - z) ** 2 / a,
        "semf_pairing_MeV": delta * c["a_P"] * a ** c["k_P"],
    }
    total = sum(terms.values())
    if a < _SEMF_LIGHT:
        ctx.warn(
            f"A = {a} < {_SEMF_LIGHT}: the liquid-drop formula fits light nuclei poorly (it has no shell "
            "structure); the AME2020 binding energy is the measured one."
        )
    coeffs = ", ".join(f"{k} = {v:g}" for k, v in c.items())
    formula = (
        "r = r0 A^(1/3); E_B = a_V A - a_S A^(2/3) - a_C Z (Z - 1) / A^(1/3) - a_A (N - Z)^2 / A "
        f"+ delta a_P A^k_P (delta = +1 even-even, -1 odd-odd, 0 odd A), {coeffs} (MeV)"
    )
    out = {
        "r0_fm": r0 * 1e15,
        "radius_fm": r0 * 1e15 * a ** (1 / 3),
        **terms,
        "semf_binding_MeV": total,
        "semf_binding_per_nucleon_MeV": total / a,
    }
    return out, formula


@op("nuclide_data", ("nuclide", "Z", "N", "r0", "semf"))
def _nuclide_data(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    if p.get("Z") is not None or p.get("N") is not None:
        if p.get("nuclide") is not None:
            one_of(p, ("nuclide", "Z"), "the nucleus (nuclide, or Z with N)")
        zz, nn = integer(p, "Z", 1), integer(p, "N", 0)
        found = _ground_by_za().get((zz, zz + nn))
        if found is None:
            model, formula = _liquid_drop(zz, nn, p, ctx)
            ctx.warn(
                f"Z = {zz}, N = {nn} is not in AME2020: only the liquid-drop radius and binding energy."
            )
            ctx.formula = formula
            ctx.formatted = (
                f"Z = {zz}, N = {nn}: radius {fmt_sig(model['radius_fm'], 5)} fm, liquid-drop B/A = "
                f"{fmt_sig(model['semf_binding_per_nucleon_MeV'], 5)} MeV."
            )
            ctx.units = units_for(model, _MODEL_UNITS)
            return {"Z": zz, "N": nn, "A": zz + nn, **model}
        key = found
    else:
        key = nuclide_key(p.get("nuclide"))
    r = _rec(key)
    z, a = r[I_Z], r[I_A]
    me, dme, est = mass_excess(key)
    h, n_ = _rec("H-1"), _rec("n-1")
    b_kev = z * Decimal(h[I_ME]) + (a - z) * Decimal(n_[I_ME]) - Decimal(r[I_ME])
    if r[I_AM]:
        am_text = str(Decimal(r[I_AM]) / Decimal(10**6))
        am = float(am_text)
        dam = float(r[I_DAM] or 0) / 1e6
    else:
        am = a + me / U_KEV
        am_text = None
        dam = dme / U_KEV
    t_half, t_text, approx = half_life(r)
    modes = [m.strip() for m in r[I_BR].split(";") if m.strip()]
    abundance = next((m for m in modes if m.startswith("IS=")), None)
    out: dict[str, Any] = {
        "nuclide": key,
        "Z": z,
        "N": a - z,
        "A": a,
        "symbol": key.split("-")[0],
        "isomer": bool(r[I_EXC]),
        "mass_excess_keV": me,
        "mass_excess_unc_keV": dme,
        "estimated": bool(est),
        "atomic_mass_u": am,
        "atomic_mass_text": am_text,
        "atomic_mass_unc_u": dam,
        "atomic_mass_kg": am * U_KG,
        "nuclear_mass_MeV": nuclear_mass_kev(key) / 1e3,
        "binding_energy_MeV": float(b_kev) / 1e3,
        "binding_per_nucleon_MeV": float(b_kev) / a / 1e3,
        "excitation_keV": float(r[I_EXC]) if r[I_EXC] else None,
        "stable": t_half == math.inf,
        "half_life_s": t_half if t_half is not None and math.isfinite(t_half) else None,
        "half_life_text": t_text,
        "spin_parity": r[I_JPI] or None,
        "decay_modes": [m for m in modes if not m.startswith("IS=")],
        "abundance_percent": float(abundance[3:].split()[0]) if abundance else None,
        "source": "AME2020 (masses), NUBASE2020 (decays); CC BY 3.0",
    }
    if out["half_life_s"]:
        out["decay_constant"] = math.log(2) / out["half_life_s"]
        out["mean_life_s"] = out["half_life_s"] / math.log(2)
    if est:
        ctx.warn("Mass estimated from systematics (# in the source), not measured.")
    if approx and t_half is not None and math.isfinite(t_half):
        ctx.warn(f"Approximate or limit half-life: {t_text}.")
    model, model_formula = _liquid_drop(z, a - z, p, ctx)  # D115
    out |= model
    b_mev = float(b_kev) / 1e3
    out["semf_minus_ame_MeV"] = model["semf_binding_MeV"] - b_mev
    if b_mev != 0:
        out["semf_error_percent"] = 100 * out["semf_minus_ame_MeV"] / b_mev
    ctx.formula = (
        "B = Z Delta(1H) + N Delta(n) - Delta(A, Z) (atomic mass excesses, AME2020); "
        "M = A u + Delta; M_nucleus = M - Z m_e; " + model_formula
    )
    ctx.formatted = (
        f"{key}: Z = {z}, N = {a - z}, atomic mass {am:.9f} u, B/A = "
        f"{fmt_sig(float(b_kev) / a / 1e3, 7)} MeV, half-life {t_text}."
    )
    ctx.units = units_for(
        out,
        {
            "mass_excess_keV": "keV",
            "mass_excess_unc_keV": "keV",
            "atomic_mass_u": "u",
            "atomic_mass_unc_u": "u",
            "atomic_mass_kg": "kg",
            "nuclear_mass_MeV": "MeV/c2",
            "binding_energy_MeV": "MeV",
            "binding_per_nucleon_MeV": "MeV",
            "excitation_keV": "keV",
            "half_life_s": "s",
            "decay_constant": "1/s",
            "mean_life_s": "s",
            "abundance_percent": "%",
            **_MODEL_UNITS,
        },
    )
    return out


# ------------------------------------------------------------------ radioactive_decay

_RD = {
    "law": {"nuclide", "T_half", "lambda", "tau", "N0", "A0", "m0", "M", "t", "fraction"},
    "activity": {"nuclide", "T_half", "lambda", "tau", "M", "m", "N", "A"},
    "chain": {"parent", "daughter", "T_half1", "T_half2", "N1_0", "N2_0", "A1_0", "branching", "t"},
}


def _decay_constant(
    p: dict[str, Any], ctx: Ctx, nkey: str = "nuclide", tkey: str = "T_half", extra: bool = True
) -> tuple[float, str | None]:
    keys = (nkey, tkey, "lambda", "tau") if extra else (nkey, tkey)
    k = one_of(p, keys, f"the half-life ({', '.join(keys)})")
    if k == nkey:
        key = nuclide_key(p[nkey])
        t, text, approx = half_life(_rec(key))
        if t is None:
            raise ToolError("INVALID_ARGUMENT", f"{key}: half-life {text}; pass it in {tkey}.")
        if t == math.inf:
            raise ToolError("INVALID_ARGUMENT", f"{key} is stable: it does not decay.")
        if approx:
            ctx.warn(f"Half-life of {key} approximate or limit: {text}.")
        return math.log(2) / t, key
    if k == tkey:
        return math.log(2) / need(p, tkey, "time"), None
    if k == "lambda":
        return need(p, "lambda", "decay_constant"), None
    return 1 / need(p, "tau", "time"), None


def _molar(p: dict[str, Any], key: str | None) -> float | None:
    if p.get("M") is not None:
        return need(p, "M", "molar_mass")
    return molar_mass_kg(key) if key else None


@op(
    "radioactive_decay",
    (
        "mode",
        "nuclide",
        "T_half",
        "lambda",
        "tau",
        "N0",
        "A0",
        "m0",
        "M",
        "t",
        "fraction",
        "m",
        "N",
        "A",
        "parent",
        "daughter",
        "T_half1",
        "T_half2",
        "N1_0",
        "N2_0",
        "A1_0",
        "branching",
    ),
)
def _radioactive_decay(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    mode = choice(p, "mode", tuple(_RD))
    only(p, mode, _RD[mode])
    if mode == "chain":
        return _chain(p, ctx)
    lam, key = _decay_constant(p, ctx)
    out: dict[str, Any] = {"lambda": lam, "T_half": math.log(2) / lam, "tau": 1 / lam}
    M = _molar(p, key)
    if mode == "activity":
        k = one_of(p, ("m", "N", "A"), "the quantity (m mass, N nuclei, A activity)")
        if k == "N":
            n = need(p, "N", None)
        elif k == "A":
            n = need(p, "A", "activity") / lam
        else:
            if M is None:
                raise ToolError(
                    "INVALID_ARGUMENT", "With the mass the nuclide or the molar mass M is needed."
                )
            n = need(p, "m", "mass") / M * N_A
        out |= {"N": n, "A": lam * n, "A_Ci": lam * n / 3.7e10}
        if M is not None:
            out |= {"m": n * M / N_A, "specific_activity": lam * N_A / M, "M": M}
        ctx.formula = "A = lambda N; N = m N_A / M; lambda = ln 2 / T_1/2"
        ctx.formatted = f"Activity {fmt_sig(lam * n)} Bq ({fmt_sig(lam * n / 3.7e10)} Ci)."
    else:
        tk = one_of(p, ("t", "fraction"), "the time t or the remaining fraction fraction")
        if tk == "t":
            t = need(p, "t", "time", positive=False)
            f = math.exp(-lam * t)
            decayed = -math.expm1(-lam * t)  # 1 - f without cancellation at small lambda t (D77)
        else:
            f = need(p, "fraction", None)
            if f > 1:
                raise ToolError(
                    "INVALID_ARGUMENT", "params.fraction (N/N0) must be between 0 and 1."
                )
            t = -math.log(f) / lam
            decayed = 1 - f
        out |= {
            "t": t,
            "fraction": f,
            "decayed_fraction": decayed,
            "half_lives": t * lam / math.log(2),
        }
        given = [k for k in ("N0", "A0", "m0") if p.get(k) is not None]
        if len(given) > 1:
            raise ToolError("INVALID_ARGUMENT", "Give only one of N0, A0, m0.")
        if given:
            g = given[0]
            if g == "N0":
                n0 = need(p, "N0", None)
            elif g == "A0":
                n0 = need(p, "A0", "activity") / lam
            else:
                if M is None:
                    raise ToolError(
                        "INVALID_ARGUMENT", "With m0 the nuclide or the molar mass M is needed."
                    )
                n0 = need(p, "m0", "mass") / M * N_A
            out |= {"N0": n0, "N": n0 * f, "A0": lam * n0, "A": lam * n0 * f}
            if M is not None:
                out |= {"m0": n0 * M / N_A, "m": n0 * f * M / N_A}
        ctx.formula = "N = N0 exp(-lambda t); lambda = ln 2 / T_1/2; t = -ln(N/N0) / lambda"
        ctx.formatted = f"After t = {fmt_sig(t)} s the remaining fraction is {fmt_sig(f, 6)}."
    if key:
        out["nuclide"] = key
    ctx.units = units_for(
        out,
        {
            "lambda": "1/s",
            "T_half": "s",
            "tau": "s",
            "t": "s",
            "fraction": "1",
            "decayed_fraction": "1",
            "half_lives": "1",
            "N0": "1",
            "N": "1",
            "A0": "Bq",
            "A": "Bq",
            "A_Ci": "Ci",
            "m0": "kg",
            "m": "kg",
            "M": "kg/mol",
            "specific_activity": "Bq/kg",
        },
    )
    return out


def _chain(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    l1, k1 = _decay_constant(p, ctx, "parent", "T_half1", extra=False)
    l2, k2 = _decay_constant(p, ctx, "daughter", "T_half2", extra=False)
    b = qty(p, "branching", None, default=1.0) or 1.0
    if b > 1:
        raise ToolError("INVALID_ARGUMENT", "params.branching is a fraction between 0 and 1.")
    k = one_of(p, ("N1_0", "A1_0"), "the initial quantity of the parent (N1_0 or A1_0)")
    n10 = need(p, "N1_0", None) if k == "N1_0" else need(p, "A1_0", "activity") / l1
    n20 = qty(p, "N2_0", None, positive=False, default=0.0) or 0.0
    t = need(p, "t", "time", positive=False)
    e1, e2 = math.exp(-l1 * t), math.exp(-l2 * t)
    n1 = n10 * e1
    d = l2 - l1
    if d == 0:
        n2 = b * n10 * l1 * t * e1 + n20 * e2
    else:
        # e1 - e2 = e1 (1 - exp(-d t)): expm1 keeps the digits when the half-lives are close (D77)
        diff = e1 * -math.expm1(-d * t) if abs(d * t) < 1 else e1 - e2
        n2 = b * n10 * l1 / d * diff + n20 * e2
    out: dict[str, Any] = {
        "lambda1": l1,
        "lambda2": l2,
        "N1": n1,
        "N2": n2,
        "A1": l1 * n1,
        "A2": l2 * n2,
        "activity_ratio": l2 * n2 / (l1 * n1) if n1 > 0 else None,
    }
    if l2 > l1:
        limit = b * l2 / (l2 - l1)
        out["equilibrium"] = "secular" if l1 / l2 < 0.01 else "transient"
        out["equilibrium_ratio"] = limit
    else:
        out["equilibrium"] = "none"
    if n20 == 0 and d != 0:
        # ln(l2 / l1) = log1p(d / l1), with d exact when the half-lives are close (D77)
        out["t_max_daughter"] = (math.log1p(d / l1) if abs(d) < 0.5 * l1 else math.log(l2 / l1)) / d
    if k1:
        out["parent"] = k1
    if k2:
        out["daughter"] = k2
    ctx.formula = (
        "N1 = N1_0 exp(-l1 t); N2 = b N1_0 l1/(l2 - l1) (exp(-l1 t) - exp(-l2 t)) + N2_0 exp(-l2 t) "
        "(Bateman); t_max = ln(l2/l1)/(l2 - l1)"
    )
    ctx.formatted = (
        f"After {fmt_sig(t)} s: parent {fmt_sig(n1)} nuclei, daughter {fmt_sig(n2)} nuclei, "
        f"activities {fmt_sig(l1 * n1)} and {fmt_sig(l2 * n2)} Bq."
    )
    ctx.units = units_for(
        out,
        {
            "lambda1": "1/s",
            "lambda2": "1/s",
            "N1": "1",
            "N2": "1",
            "A1": "Bq",
            "A2": "Bq",
            "activity_ratio": "1",
            "equilibrium_ratio": "1",
            "t_max_daughter": "s",
        },
    )
    return out


# ------------------------------------------------------------------ nuclear_reaction

# name -> (canonical name, charge, lepton number, mass in keV)
_LEPTONS: dict[str, tuple[str, int, int, float]] = {}
for _names, _val in (
    (("e-", "electron", "beta-", "β-"), ("e-", -1, 1, ME_KEV)),
    (("e⁺", "positron", "beta⁺", "β⁺"), ("e+", 1, -1, ME_KEV)),
    (("nu", "ν", "nu_e", "neutrino"), ("nu", 0, 1, 0.0)),
    (("anti-nu", "antinu", "nubar", "ν̄", "antineutrino"), ("anti-nu", 0, -1, 0.0)),
    (("gamma", "γ"), ("gamma", 0, 0, 0.0)),
):
    for _n in _names:
        _LEPTONS[_n] = _val
_ARROWS = ("-->", "->", "→", "=>", "=")


def _term(raw: str) -> tuple[int, str, bool]:
    """(coefficient, canonical name, is a nucleus)."""
    t = raw.strip()
    if not t:
        raise ToolError("INVALID_ARGUMENT", "Empty species in the reaction: check the '+'.")
    cands = [(1, t)]
    m = re.match(r"^(\d+)\s*(\S.*)$", t)
    if m:
        cands.append((int(m.group(1)), m.group(2)))
    if " " in t and m:  # "2 p": the coefficient first
        cands.reverse()
    for coef, name in cands:
        if name in _LEPTONS:
            return coef, _LEPTONS[name][0], False
        key = _try_nuclide(name)
        if key is not None:
            return coef, key, True
    nuclide_key(cands[-1][1])  # error with the suggestion
    raise AssertionError  # pragma: no cover


def _side(text: str) -> list[tuple[int, str, bool]]:
    return [_term(x) for x in text.split("+")]


def _parts(item: tuple[int, str, bool]) -> tuple[float, int, int, int, float, bool, float]:
    """(weight in keV for Q, charge, A, lepton number, uncertainty keV, estimated, mass keV)."""
    _, name, nucleus = item
    if nucleus:
        r = _rec(name)
        me, dme, est = mass_excess(name)
        return me - r[I_Z] * ME_KEV, r[I_Z], r[I_A], 0, dme, est, nuclear_mass_kev(name)
    lep = next(v for v in _LEPTONS.values() if v[0] == name)
    return lep[3], lep[1], 0, lep[2], 0.0, False, lep[3]


@op("nuclear_reaction", ("reaction", "projectile"))
def _nuclear_reaction(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    text = p.get("reaction")
    if not isinstance(text, str) or not 0 < len(text) <= 300:
        raise ToolError(
            "INVALID_ARGUMENT", 'params.reaction: text like "U-235 + n -> Ba-141 + Kr-92 + 3 n".'
        )
    text = re.sub(r"(?<![A-Za-z0-9])(e|beta|β)\+", r"\1⁺", text)
    arrow = next((a for a in _ARROWS if a in text), None)
    if arrow is None:
        raise ToolError(
            "INVALID_ARGUMENT", "The reaction arrow (->) is missing.", '"H-2 + H-3 -> He-4 + n"'
        )
    left, _, right = text.partition(arrow)
    reac, prod = _side(left), _side(right)

    def tot(items: list[tuple[int, str, bool]], i: int) -> float:
        return sum(it[0] * _parts(it)[i] for it in items)

    a_r, a_p = tot(reac, 2), tot(prod, 2)
    z_r, z_p = tot(reac, 1), tot(prod, 1)
    if a_r != a_p:
        raise ToolError(
            "INVALID_ARGUMENT", f"Nucleon number not conserved: {a_r:g} against {a_p:g}."
        )
    if z_r != z_p:
        raise ToolError(
            "INVALID_ARGUMENT",
            f"Charge not conserved: {z_r:g} against {z_p:g} (count e- and e+ too).",
        )
    if tot(reac, 3) != tot(prod, 3):
        ctx.warn("Lepton number not conserved: is a neutrino or an antineutrino missing?")
    q = tot(reac, 0) - tot(prod, 0)
    dq = math.sqrt(sum((it[0] * _parts(it)[4]) ** 2 for it in reac + prod))
    if any(_parts(it)[5] for it in reac + prod):
        ctx.warn("At least one mass is estimated from systematics (# in the source).")

    def label(items: list[tuple[int, str, bool]]) -> str:
        return " + ".join(f"{c} {n}" if c > 1 else n for c, n, _ in items)

    out: dict[str, Any] = {
        "reaction": f"{label(reac)} -> {label(prod)}",
        "Q_MeV": q / 1e3,
        "Q_keV": q,
        "Q_J": q * KEV_J,
        "Q_unc_keV": dq,
        "mass_defect_u": q / U_KEV,
        "exothermic": q > 0,
    }
    formulas = [
        "Q = sum(M_nucleus + m_leptons)_reactants - sum(...)_products, M_nucleus = A u + Delta - Z m_e"
    ]
    m_r, m_p = tot(reac, 6), tot(prod, 6)
    n_r = sum(c for c, _, _ in reac)
    n_p = sum(c for c, _, _ in prod)
    if q > 0:
        mass_kg = sum(c * molar_mass_kg(n) / M_U * U_KG for c, n, nuc in reac if nuc)
        if mass_kg > 0:
            out["energy_per_kg_reactants_J"] = q * KEV_J / mass_kg
    if q < 0 and n_r == 2:
        names = [n for c, n, _ in reac for _ in range(c)]
        proj = p.get("projectile")
        if proj is not None:
            pk = (
                _LEPTONS[proj][0]
                if isinstance(proj, str) and proj in _LEPTONS
                else nuclide_key(proj)
            )
            if pk not in names:
                raise ToolError(
                    "INVALID_ARGUMENT", f"params.projectile: '{proj}' is not among the reactants."
                )
            names.remove(pk)
        else:
            names = names[1:]
        target = next(it for it in reac if it[1] == names[0])
        m_t = _parts(target)[6]
        out["T_threshold_MeV"] = -q * (m_r + m_p) / (2 * m_t) / 1e3
        out["T_threshold_classical_MeV"] = -q * (1 + (m_r - m_t) / m_t) / 1e3
        formulas.append("threshold T = -Q (M_initial + M_final) / (2 M_target), target at rest")
    elif p.get("projectile") is not None:
        raise ToolError(
            "INVALID_ARGUMENT",
            "params.projectile is used only for the threshold (Q < 0, two reactants).",
        )
    if q > 0 and n_r == 1 and n_p == 2:
        ms = [_parts(it)[6] for it in prod for _ in range(it[0])]
        big = m_r
        energies = [(big**2 + ms[i] ** 2 - ms[1 - i] ** 2) / (2 * big) - ms[i] for i in (0, 1)]
        out["T_products_MeV"] = [e / 1e3 for e in energies]
        formulas.append("two-body decay: E_i = (M^2 + m_i^2 - m_j^2) / (2 M), T_i = E_i - m_i")
    ctx.formula = "; ".join(formulas)
    ctx.formatted = f"Q = {fmt_sig(q / 1e3, 7)} MeV ({'exothermic' if q > 0 else 'endothermic'})."
    ctx.units = units_for(
        out,
        {
            "Q_MeV": "MeV",
            "Q_keV": "keV",
            "Q_J": "J",
            "Q_unc_keV": "keV",
            "mass_defect_u": "u",
            "energy_per_kg_reactants_J": "J/kg",
            "T_threshold_MeV": "MeV",
            "T_threshold_classical_MeV": "MeV",
            "T_products_MeV": "MeV",
        },
    )
    return out
