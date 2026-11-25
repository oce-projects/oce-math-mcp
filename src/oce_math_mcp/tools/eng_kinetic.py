"""eng_calc: kinetic theory of gases (D87).

Sources: Wikipedia pages "Kinetic theory of gases", "Maxwell-Boltzmann distribution", "Mean free path" and
"Graham's law" in the local offline library.
- Speeds of the Maxwell-Boltzmann distribution: v_p = sqrt(2 k T / m), <v> = sqrt(8 k T / (pi m)),
  v_rms = sqrt(3 k T / m), mean relative speed sqrt(2) <v>; mean translational energy 3/2 k T per molecule;
  speed of sound c = sqrt(gamma / 3) v_rms with gamma = 1 + 2 / f (f degrees of freedom).
- Speed distribution: f(v) = sqrt(2 / pi) v^2 / a^3 exp(-v^2 / (2 a^2)), a = sqrt(k T / m) (chi distribution
  with 3 degrees of freedom); cumulative F(v) = erf(x / sqrt 2) - sqrt(2 / pi) x exp(-x^2 / 2), x = v / a. The
  fraction above v is computed as erfc(x / sqrt 2) + sqrt(2 / pi) x exp(-x^2 / 2), without the cancellation of
  1 - F far in the tail.
- Pressure p = rho <v^2> / 3; with the molecule mass T = m <v^2> / (3 k).
- Mean free path l = 1 / (sqrt 2 n sigma) = k T / (sqrt 2 pi d^2 p); hard spheres with the same viscosity:
  l = (mu / p) sqrt(pi k T / (2 m)). Collision frequency z = <v> / l.
- Wall (impingement) flux J = n <v> / 4 = p / sqrt(2 pi m k T); effusion through a small hole Phi = J A.
- Graham's law: rate_1 / rate_2 = sqrt(M_2 / M_1); M_2 = (rate_1 / rate_2)^2 M_1.
- Equipartition: 1/2 k T per kinetic degree of freedom; monatomic f = 3, diatomic 5, polyatomic 6 (the page:
  vibrational modes need quantum statistics and are not included unless counted in f).
"""

from __future__ import annotations

import math
from typing import Any

from ..constants import lookup
from ..envelope import Ctx
from ..errors import ToolError
from .eng_base import choice, fmt_sig, integer, need, one_of, only, op, temperature, units_for

K_B = lookup("Boltzmann constant")[1]
N_A = lookup("Avogadro constant")[1]
R_MOLAR = K_B * N_A
_SQRT_2_PI = math.sqrt(2 / math.pi)

_MOLECULE = {"m", "M_molar"}
_MODES: dict[str, set[str]] = {
    "speeds": {"T", "dof"} | _MOLECULE,
    "pressure": {"rho", "v_rms"} | _MOLECULE,
    "distribution": {"T", "v", "v1", "v2"} | _MOLECULE,
    "mean_free_path": {"T", "p", "d", "sigma", "mu", "L"} | _MOLECULE,
    "wall_flux": {"T", "p", "A"} | _MOLECULE,
    "graham": {"M1", "M2", "rate_ratio"},
    "equipartition": {"dof", "gas", "T", "n_mol"},
}
_GAS_DOF = {"monatomic": 3, "diatomic": 5, "polyatomic": 6}
_UNITS = {
    "m": "kg",
    "M_molar": "kg/mol",
    "v_p": "m/s",
    "v_mean": "m/s",
    "v_rms": "m/s",
    "v_rel": "m/s",
    "E_k": "J",
    "E_k_molar": "J/mol",
    "gamma": "1",
    "c_sound": "m/s",
    "p": "Pa",
    "T": "K",
    "n": "1/m3",
    "K_t_density": "J/m3",
    "a": "m/s",
    "pdf": "s/m",
    "cdf": "1",
    "fraction_above": "1",
    "fraction_between": "1",
    "sigma": "m2",
    "ell": "m",
    "z": "1/s",
    "tau": "s",
    "Kn": "1",
    "J": "1/(m2*s)",
    "Phi": "1/s",
    "mdot": "kg/s",
    "rate_ratio": "1",
    "time_ratio": "1",
    "M1": "kg/mol",
    "M2": "kg/mol",
    "dof": "1",
    "Cv_molar": "J/(mol*K)",
    "Cp_molar": "J/(mol*K)",
    "U_molar": "J/mol",
    "E_molecule": "J",
    "C_V": "J/K",
    "C_p": "J/K",
    "U": "J",
}


def _molecule(p: dict[str, Any]) -> float:
    """Mass of one molecule in kg, from m or from the molar mass M_molar."""
    if (
        one_of(p, ("m", "M_molar"), "the molecule mass (m) or the molar mass (M_molar) of the gas")
        == "m"
    ):
        return need(p, "m", "mass")
    return need(p, "M_molar", "molar_mass") / N_A


def _kelvin(p: dict[str, Any]) -> float:
    t = temperature(p, "T")
    assert t is not None
    return t


def _speeds(m: float, t: float) -> dict[str, float]:
    kt_m = K_B * t / m
    mean = math.sqrt(8 * kt_m / math.pi)
    return {
        "v_p": math.sqrt(2 * kt_m),
        "v_mean": mean,
        "v_rms": math.sqrt(3 * kt_m),
        "v_rel": math.sqrt(2) * mean,
    }


@op("kinetic_theory", ("mode", *sorted(set().union(*_MODES.values()))))
def _kinetic(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    mode = choice(p, "mode", tuple(_MODES))
    only(p, mode, _MODES[mode])
    out = {
        "speeds": _mode_speeds,
        "pressure": _mode_pressure,
        "distribution": _mode_distribution,
        "mean_free_path": _mode_mfp,
        "wall_flux": _mode_flux,
        "graham": _mode_graham,
        "equipartition": _mode_equipartition,
    }[mode](p, ctx)
    ctx.units = units_for(out, _UNITS)
    return out


def _mode_speeds(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    m, t = _molecule(p), _kelvin(p)
    out: dict[str, Any] = {"m": m, "M_molar": m * N_A, **_speeds(m, t)}
    out["E_k"] = 1.5 * K_B * t
    out["E_k_molar"] = 1.5 * R_MOLAR * t
    ctx.formula = (
        "v_p = sqrt(2 k T / m), <v> = sqrt(8 k T / (pi m)), v_rms = sqrt(3 k T / m), v_rel = sqrt(2) <v>; "
        "E_k = 3/2 k T"
    )
    if p.get("dof") is not None:
        f = integer(p, "dof", 1)
        out["gamma"] = 1 + 2 / f
        out["c_sound"] = math.sqrt(out["gamma"] / 3) * out["v_rms"]
        ctx.formula += "; gamma = 1 + 2 / f, c = sqrt(gamma / 3) v_rms"
    ctx.formatted = (
        f"Molecular speeds at {fmt_sig(t, 6)} K: most probable {fmt_sig(out['v_p'], 6)} m/s, "
        f"mean {fmt_sig(out['v_mean'], 6)} m/s, rms {fmt_sig(out['v_rms'], 6)} m/s."
    )
    return out


def _mode_pressure(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    rho, v_rms = need(p, "rho", "density"), need(p, "v_rms", "velocity")
    pressure = rho * v_rms * v_rms / 3
    out: dict[str, Any] = {"p": pressure, "K_t_density": 1.5 * pressure}
    ctx.formula = "p = rho <v^2> / 3; translational energy per volume 3/2 p"
    if p.get("m") is not None or p.get("M_molar") is not None:
        m = _molecule(p)
        out["T"] = m * v_rms * v_rms / (3 * K_B)
        out["n"] = rho / m
        ctx.formula += "; T = m <v^2> / (3 k), n = rho / m"
    ctx.formatted = f"Kinetic pressure p = {fmt_sig(pressure, 6)} Pa."
    return out


def _cdf_sf(x: float) -> tuple[float, float]:
    """Cumulative fraction and fraction above for x = v / a (chi distribution, 3 degrees of freedom)."""
    g = _SQRT_2_PI * x * math.exp(-x * x / 2)
    return math.erf(x / math.sqrt(2)) - g, math.erfc(x / math.sqrt(2)) + g


def _mode_distribution(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    m, t = _molecule(p), _kelvin(p)
    a = math.sqrt(K_B * t / m)
    out: dict[str, Any] = {"a": a, "v_p": math.sqrt(2) * a}
    ctx.formula = "f(v) = sqrt(2 / pi) v^2 / a^3 exp(-v^2 / (2 a^2)), a = sqrt(k T / m)"
    has_v = p.get("v") is not None
    has_band = p.get("v1") is not None or p.get("v2") is not None
    if not has_v and not has_band:
        raise ToolError(
            "INVALID_ARGUMENT",
            "Give v (density and cumulative fraction at that speed) or v1 and v2 (fraction between them).",
        )
    if has_v:
        v = need(p, "v", "velocity", positive=False)
        if v < 0:
            raise ToolError("INVALID_ARGUMENT", "params.v (a speed) must be >= 0.")
        x = v / a
        out["pdf"] = _SQRT_2_PI * x * x / a * math.exp(-x * x / 2)
        out["cdf"], out["fraction_above"] = _cdf_sf(x)
        ctx.formula += "; F(v) = erf(x / sqrt 2) - sqrt(2 / pi) x exp(-x^2 / 2), x = v / a"
    if has_band:
        v1 = need(p, "v1", "velocity", positive=False)
        v2 = need(p, "v2", "velocity")
        if not 0 <= v1 < v2:
            raise ToolError("INVALID_ARGUMENT", "params.v1 and v2: 0 <= v1 < v2 is needed.")
        (c1, s1), (c2, s2) = _cdf_sf(v1 / a), _cdf_sf(v2 / a)
        # the smaller of the two differences keeps the digits (lower part or tail of the distribution)
        out["fraction_between"] = c2 - c1 if c1 < 0.5 else s1 - s2
        ctx.formula += "; fraction between v1 and v2 = F(v2) - F(v1)"
    shown = out["fraction_between"] if has_band else out["cdf"]
    ctx.formatted = (
        f"Maxwell-Boltzmann speeds at {fmt_sig(t, 6)} K (a = {fmt_sig(a, 6)} m/s): "
        f"fraction {fmt_sig(shown, 6)}."
    )
    return out


def _mode_mfp(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    t, pressure = _kelvin(p), need(p, "p", "pressure")
    n = pressure / (K_B * t)
    out: dict[str, Any] = {"n": n}
    how = one_of(
        p, ("d", "sigma", "mu"), "the size of the molecules (d, sigma or the viscosity mu)"
    )
    if how == "mu":
        m = _molecule(p)
        out["ell"] = (
            need(p, "mu", "dynamic_viscosity") / pressure * math.sqrt(math.pi * K_B * t / (2 * m))
        )
        ctx.formula = "l = (mu / p) sqrt(pi k T / (2 m)) (hard spheres with the same viscosity)"
    else:
        sigma = math.pi * need(p, "d", "length") ** 2 if how == "d" else need(p, "sigma", "area")
        out["sigma"] = sigma
        out["ell"] = 1 / (math.sqrt(2) * n * sigma)
        ctx.formula = (
            "n = p / (k T), sigma = pi d^2, l = 1 / (sqrt 2 n sigma) = k T / (sqrt 2 pi d^2 p)"
        )
    if p.get("m") is not None or p.get("M_molar") is not None:
        out["v_mean"] = _speeds(_molecule(p), t)["v_mean"]
        out["z"] = out["v_mean"] / out["ell"]
        out["tau"] = 1 / out["z"]
        ctx.formula += "; z = <v> / l, tau = 1 / z"
    if p.get("L") is not None:
        out["Kn"] = out["ell"] / need(p, "L", "length")
        ctx.formula += "; Kn = l / L"
    ctx.formatted = f"Mean free path l = {fmt_sig(out['ell'], 6)} m at {fmt_sig(pressure, 6)} Pa."
    return out


def _mode_flux(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    m, t, pressure = _molecule(p), _kelvin(p), need(p, "p", "pressure")
    flux = pressure / math.sqrt(2 * math.pi * m * K_B * t)
    out: dict[str, Any] = {
        "n": pressure / (K_B * t),
        "v_mean": _speeds(m, t)["v_mean"],
        "J": flux,
    }
    ctx.formula = "J = n <v> / 4 = p / sqrt(2 pi m k T)"
    if p.get("A") is not None:
        out["Phi"] = flux * need(p, "A", "area")
        out["mdot"] = out["Phi"] * m
        ctx.formula += "; effusion through a small hole Phi = J A, mass flow m Phi"
    ctx.formatted = f"Wall flux J = {fmt_sig(flux, 6)} molecules per m2 per s."
    return out


def _mode_graham(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    m1 = need(p, "M1", "molar_mass")
    if (
        one_of(p, ("M2", "rate_ratio"), "the second gas (M2 or rate_ratio = rate_1 / rate_2)")
        == "M2"
    ):
        m2 = need(p, "M2", "molar_mass")
        ratio = math.sqrt(m2 / m1)
    else:
        ratio = need(p, "rate_ratio", None)
        m2 = ratio * ratio * m1
    ctx.formula = "rate_1 / rate_2 = sqrt(M2 / M1); M2 = (rate_1 / rate_2)^2 M1"
    ctx.formatted = (
        f"Graham: gas 1 effuses {fmt_sig(ratio, 6)} times as fast as gas 2 "
        f"(M2 = {fmt_sig(m2 * 1000, 6)} g/mol)."
    )
    return {"M1": m1, "M2": m2, "rate_ratio": ratio, "time_ratio": 1 / ratio}


def _mode_equipartition(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    if one_of(p, ("gas", "dof"), "the degrees of freedom (gas or dof)") == "gas":
        f = _GAS_DOF[choice(p, "gas", tuple(_GAS_DOF))]
    else:
        f = integer(p, "dof", 1)
    out: dict[str, Any] = {
        "dof": f,
        "Cv_molar": f / 2 * R_MOLAR,
        "Cp_molar": (f + 2) / 2 * R_MOLAR,
        "gamma": (f + 2) / f,
    }
    ctx.formula = (
        "Cv = f/2 R, Cp = (f + 2)/2 R, gamma = (f + 2) / f (classical, without vibrational modes)"
    )
    t = temperature(p, "T", need=False)
    if t is not None:
        out["U_molar"] = f / 2 * R_MOLAR * t
        out["E_molecule"] = f / 2 * K_B * t
        ctx.formula += "; U = f/2 R T per mole, f/2 k T per molecule"
    if p.get("n_mol") is not None:
        n_mol = need(p, "n_mol", "amount_of_substance")
        out["C_V"], out["C_p"] = n_mol * out["Cv_molar"], n_mol * out["Cp_molar"]
        if t is not None:
            out["U"] = n_mol * out["U_molar"]
    ctx.formatted = (
        f"Equipartition with {f} degrees of freedom: Cv = {fmt_sig(out['Cv_molar'], 6)} J/(mol K), "
        f"gamma = {fmt_sig(out['gamma'], 6)}."
    )
    return out
