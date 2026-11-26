"""eng_calc: statistical mechanics (D98).

Source: Wikipedia "Table of thermodynamic equations", sections "Statistical physics" and "Quantum properties", in
the local offline library; "Maxwell-Juttner distribution" and "Sackur-Tetrode equation" for the checks.
- Entropy: Gibbs S = -k sum P_i ln P_i; Boltzmann S = k ln Omega, the same when P_i = 1 / Omega. Also in bits
  (S / (k ln 2)) and the largest entropy of n states, k ln n.
- Entropy from heat: dS = integral dQ / T; isothermal Q / T; constant heat capacity C ln(T2 / T1). With a
  reservoir at T_r the entropy generated is dS - Q / T_r (second law: >= 0).
- Maxwell-Juttner (relativistic ideal gas): theta = k T / (m c^2); f(p) = exp(-gamma(p) / theta) /
  (4 pi m^3 c^3 theta K2(1 / theta)) with gamma(p) = sqrt(1 + (p / m c)^2); density in gamma
  gamma^2 beta exp(-gamma / theta) / (theta K2(1 / theta)); <gamma> = K1(1 / theta) / K2(1 / theta) + 3 theta.
  Computed with mpmath: K2(1 / theta) underflows in floats for a cold gas, and <gamma> - 1 cancels.
- Partition functions of one molecule (high-temperature forms of the page): translational
  Z_t = (2 pi m k T)^(3/2) V / h^3 = V / Lambda^3, vibrational Z_v = 1 / (1 - exp(-h nu / k T)) (energy zero at
  the ground state), rotational Z_r = 2 I k T / (sigma hbar^2). For N molecules U = N k T^2 d ln Z / dT and
  S = U / T + N k ln Z - N k ln N + N k (indistinguishable molecules, Stirling); A = U - T S.
"""

from __future__ import annotations

import math
from typing import Any

import mpmath

from ..constants import lookup
from ..envelope import Ctx
from ..errors import ToolError
from ..limits import LIST_MAX
from .eng_base import (
    choice,
    fmt_sig,
    integer,
    log_ratio,
    need,
    one_of,
    only,
    op,
    qty,
    range_guard,
    temperature,
    units_for,
)

K_B = lookup("Boltzmann constant")[1]
H = lookup("Planck constant")[1]
HBAR = lookup("reduced Planck constant")[1]
C_LIGHT = lookup("speed of light in vacuum")[1]
N_A = lookup("Avogadro constant")[1]
_LN2 = math.log(2)
_ROT_WARN = (
    10.0  # the classical Z_r is off by about theta_rot / (3 T): over 3% below T = 10 theta_rot
)

_MODES: dict[str, set[str]] = {
    "entropy": {"probabilities", "omega", "ln_omega"},
    "entropy_heat": {"Q", "T", "C", "T1", "T2", "T_reservoir"},
    "maxwell_juttner": {"T", "m", "p", "gamma"},
    "partition_functions": {"T", "m", "M_molar", "V", "nu", "omega", "I", "sigma", "N", "n_mol"},
}
_UNITS = {
    "S": "J/K",
    "S_over_k": "1",
    "S_bits": "1",
    "S_max_over_k": "1",
    "n_states": "1",
    "dS": "J/K",
    "Q": "J",
    "dS_reservoir": "J/K",
    "S_generated": "J/K",
    "theta": "1",
    "gamma_mean": "1",
    "E_mean": "J",
    "E_kin_mean": "J",
    "E_kin_classical": "J",
    "gamma_p": "1",
    "pdf_p": "s3/(kg3*m3)",
    "pdf_p_magnitude": "s/(kg*m)",
    "pdf_gamma": "1",
    "Lambda": "m",
    "Z_t": "1",
    "Z_v": "1",
    "Z_r": "1",
    "Z": "1",
    "ln_Z": "1",
    "theta_vib": "K",
    "theta_rot": "K",
    "N": "1",
    "U_trans": "J",
    "U_vib": "J",
    "U_rot": "J",
    "U": "J",
    "C_V": "J/K",
    "A": "J",
}


@op("statistical_mechanics", ("mode", *sorted(set().union(*_MODES.values()))))
@range_guard
def _statistical(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    mode = choice(p, "mode", tuple(_MODES))
    only(p, mode, _MODES[mode])
    out = {
        "entropy": _mode_entropy,
        "entropy_heat": _mode_entropy_heat,
        "maxwell_juttner": _mode_juttner,
        "partition_functions": _mode_partition,
    }[mode](p, ctx)
    ctx.units = units_for(out, _UNITS)
    return out


def _probabilities(raw: Any) -> list[float]:
    if not isinstance(raw, list) or not raw:
        raise ToolError(
            "INVALID_ARGUMENT", "params.probabilities is a non-empty list of numbers.", "[0.5, 0.5]"
        )
    if len(raw) > LIST_MAX:
        raise ToolError("LIMIT_EXCEEDED", f"At most {LIST_MAX} probabilities.")
    probs: list[float] = []
    for v in raw:
        if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(float(v)):
            raise ToolError("INVALID_ARGUMENT", "Every probability must be a finite number.")
        if v < 0 or v > 1:
            raise ToolError("INVALID_ARGUMENT", "Every probability must be between 0 and 1.")
        probs.append(float(v))
    total = math.fsum(probs)
    if abs(total - 1) > 1e-9:
        raise ToolError(
            "INVALID_ARGUMENT", f"The probabilities must add up to 1 (sum = {total!r})."
        )
    return probs


def _mode_entropy(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    key = one_of(p, ("probabilities", "omega", "ln_omega"), "the entropy")
    out: dict[str, Any] = {}
    if key == "probabilities":
        probs = _probabilities(p["probabilities"])
        s_k = -math.fsum(x * math.log(x) for x in probs if x > 0)
        out["n_states"] = len(probs)
        out["S_max_over_k"] = math.log(len(probs))
        out["uniform"] = max(probs) - min(probs) <= 1e-12 * max(probs)
        ctx.formula = "S = -k sum P_i ln P_i; with P_i = 1 / Omega for all states S = k ln Omega"
    elif key == "omega":
        omega = need(p, "omega", None)
        if omega < 1:
            raise ToolError(
                "INVALID_ARGUMENT", "params.omega (number of microstates) must be >= 1."
            )
        s_k = math.log(omega)
        ctx.formula = "S = k ln Omega"
    else:
        s_k = need(p, "ln_omega", None, positive=False)
        if s_k < 0:
            raise ToolError("INVALID_ARGUMENT", "params.ln_omega must be >= 0 (Omega >= 1).")
        ctx.formula = "S = k ln Omega"
    out.update({"S": K_B * s_k, "S_over_k": s_k, "S_bits": s_k / _LN2})
    ctx.formatted = f"S = {fmt_sig(out['S'], 6)} J/K ({fmt_sig(out['S_bits'], 6)} bits)."
    return out


def _mode_entropy_heat(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    if one_of(p, ("Q", "C"), "the heat (Q at constant T, or the heat capacity C in J/K)") == "Q":
        if p.get("T1") is not None or p.get("T2") is not None:
            raise ToolError("INVALID_ARGUMENT", "With Q the temperature is T (isothermal heat).")
        q = need(p, "Q", "energy", positive=False)
        t = temperature(p, "T")
        assert t is not None
        ds = q / t
        ctx.formula = "dS = Q / T (heat at constant temperature, reversible)"
    else:
        if p.get("T") is not None:
            raise ToolError("INVALID_ARGUMENT", "With C the temperatures are T1 and T2.")
        cap = need(p, "C", None)
        t1, t2 = temperature(p, "T1"), temperature(p, "T2")
        assert t1 is not None and t2 is not None
        ds = cap * log_ratio(t2, t1)
        q = cap * (t2 - t1)
        ctx.formula = "dS = integral C dT / T = C ln(T2 / T1), Q = C (T2 - T1)"
    out: dict[str, Any] = {"dS": ds, "Q": q}
    t_res = temperature(p, "T_reservoir", need=False)
    if t_res is not None:
        out["dS_reservoir"] = -q / t_res
        out["S_generated"] = ds - q / t_res
        ctx.formula += "; reservoir dS_r = -Q / T_r, generated S_gen = dS + dS_r"
        if out["S_generated"] < -1e-12 * abs(ds):
            ctx.warn(
                "S_generated < 0: this heat exchange with the reservoir is impossible (second law)."
            )
    ctx.formatted = f"dS = {fmt_sig(ds, 6)} J/K."
    return out


def _mode_juttner(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    t = temperature(p, "T")
    assert t is not None
    m = need(p, "m", "mass")
    rest = m * C_LIGHT * C_LIGHT
    theta = K_B * t / rest
    # <gamma> - 1 loses about -log10(theta) digits: the working precision follows
    dps = 40 + max(0, int(-math.log10(theta)))
    with mpmath.workdps(dps):
        th = mpmath.mpf(theta)
        x = 1 / th
        k2 = mpmath.besselk(2, x)
        g_mean = mpmath.besselk(1, x) / k2 + 3 * th
        out: dict[str, Any] = {
            "theta": theta,
            "gamma_mean": float(g_mean),
            "E_mean": float(g_mean * rest),
            "E_kin_mean": float((g_mean - 1) * rest),
            "E_kin_classical": 1.5 * K_B * t,
        }
        mc = mpmath.mpf(m) * C_LIGHT
        if p.get("p") is not None:
            mom = mpmath.mpf(need(p, "p", "momentum", positive=False))
            if mom < 0:
                raise ToolError("INVALID_ARGUMENT", "params.p (momentum magnitude) must be >= 0.")
            u2 = (mom / mc) ** 2
            g_p = mpmath.sqrt(1 + u2)
            # exp(-gamma / theta) / K2(1 / theta) written with gamma - 1 = u^2 / (gamma + 1)
            f = mpmath.exp(-(u2 / (g_p + 1)) * x) / (
                4 * mpmath.pi * mc**3 * th * k2 * mpmath.exp(x)
            )
            out["gamma_p"] = float(g_p)
            out["pdf_p"] = float(f)
            out["pdf_p_magnitude"] = float(4 * mpmath.pi * mom * mom * f)
        if p.get("gamma") is not None:
            g = need(p, "gamma", None)
            if g < 1:
                raise ToolError("INVALID_ARGUMENT", "params.gamma (Lorentz factor) must be >= 1.")
            gm = mpmath.mpf(g)
            beta = mpmath.sqrt(1 - 1 / (gm * gm))
            out["pdf_gamma"] = float(
                gm * gm * beta * mpmath.exp(-(gm - 1) * x) / (th * k2 * mpmath.exp(x))
            )
    ctx.formula = (
        "theta = k T / (m c^2); f(p) = exp(-gamma(p) / theta) / (4 pi m^3 c^3 theta K2(1 / theta)); "
        "<gamma> = K1(1 / theta) / K2(1 / theta) + 3 theta"
    )
    ctx.formatted = (
        f"theta = {fmt_sig(theta, 6)}, <gamma> = {fmt_sig(out['gamma_mean'], 8)}, "
        f"mean kinetic energy {fmt_sig(out['E_kin_mean'], 6)} J."
    )
    return out


def _exp_or_none(ln_value: float) -> float | None:
    return math.exp(ln_value) if ln_value < 700 else None


def _mode_partition(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    t = temperature(p, "T")
    assert t is not None
    kt = K_B * t
    out: dict[str, Any] = {}
    ln_z = 0.0
    u_k: dict[str, float] = {}  # energy per molecule
    c_k: dict[str, float] = {}  # heat capacity per molecule in units of k
    formulas: list[str] = []
    if any(p.get(k) is not None for k in ("m", "M_molar", "V")):
        if p.get("V") is None or (p.get("m") is None and p.get("M_molar") is None):
            raise ToolError(
                "INVALID_ARGUMENT", "The translational part needs the mass (m or M_molar) and V."
            )
        if one_of(p, ("m", "M_molar"), "the molecule mass (m) or the molar mass (M_molar)") == "m":
            m = need(p, "m", "mass")
        else:
            m = need(p, "M_molar", "molar_mass") / N_A
        vol = need(p, "V", "volume")
        lam = H / math.sqrt(2 * math.pi * m * kt)
        ln_zt = math.log(vol) - 3 * math.log(lam)
        out["Lambda"] = lam
        zt = _exp_or_none(ln_zt)
        if zt is not None:
            out["Z_t"] = zt
        ln_z += ln_zt
        u_k["trans"], c_k["trans"] = 1.5 * kt, 1.5
        formulas.append("Z_t = (2 pi m k T)^(3/2) V / h^3 = V / Lambda^3")
    if p.get("nu") is not None or p.get("omega") is not None:
        if (
            one_of(p, ("nu", "omega"), "the vibration (frequency nu or angular frequency omega)")
            == "nu"
        ):
            quantum = H * need(p, "nu", "frequency")
        else:
            quantum = HBAR * need(p, "omega", "angular_velocity")
        x = quantum / kt
        one_minus = -math.expm1(-x)  # 1 - exp(-x) without cancellation for small x
        out["theta_vib"] = quantum / K_B
        out["Z_v"] = 1 / one_minus
        ln_z += -math.log(one_minus)
        u_k["vib"] = quantum * math.exp(-x) / one_minus
        c_k["vib"] = x * x * math.exp(-x) / (one_minus * one_minus)
        formulas.append("Z_v = 1 / (1 - exp(-h nu / k T)), energy zero at the ground state")
    if p.get("I") is not None:
        inertia = need(p, "I", "mass_moment")
        sigma = integer(p, "sigma", 1) if p.get("sigma") is not None else 1
        theta_rot = HBAR * HBAR / (2 * inertia * K_B)
        out["theta_rot"] = theta_rot
        out["Z_r"] = 2 * inertia * kt / (sigma * HBAR * HBAR)
        ln_z += math.log(out["Z_r"])
        u_k["rot"], c_k["rot"] = kt, 1.0
        formulas.append("Z_r = 2 I k T / (sigma hbar^2) (linear molecule, T >> theta_rot)")
        if t < _ROT_WARN * theta_rot:
            ctx.warn(
                f"T = {fmt_sig(t, 4)} K is below 10 theta_rot ({fmt_sig(theta_rot, 4)} K): the classical Z_r "
                "is off by about theta_rot / (3 T); sum the rotational levels instead."
            )
    elif p.get("sigma") is not None:
        raise ToolError("INVALID_ARGUMENT", "params.sigma (symmetry number) goes with I.")
    if not u_k:
        raise ToolError(
            "INVALID_ARGUMENT",
            "Give at least one motion: translation (m or M_molar, and V), vibration (nu or omega), rotation (I).",
            '{"mode": "partition_functions", "T": 300, "M_molar": 0.04, "V": 0.0248, "n_mol": 1}',
        )
    out["ln_Z"] = ln_z
    z = _exp_or_none(ln_z)
    if z is not None:
        out["Z"] = z
    ctx.formula = "; ".join(formulas) + "; Z = product of the parts"
    if p.get("N") is not None or p.get("n_mol") is not None:
        if one_of(p, ("N", "n_mol"), "the number of molecules (N or n_mol)") == "N":
            n = need(p, "N", None)
        else:
            n = qty(p, "n_mol", "amount_of_substance")
            assert n is not None
            n *= N_A
        out["N"] = n
        for part, u in u_k.items():
            out[f"U_{part}"] = n * u
        out["U"] = n * math.fsum(u_k.values())
        out["C_V"] = n * K_B * math.fsum(c_k.values())
        out["S"] = out["U"] / t + n * K_B * (ln_z - math.log(n) + 1)
        out["A"] = out["U"] - t * out["S"]
        ctx.formula += (
            "; U = N k T^2 d ln Z / dT; S = U / T + N k ln Z - N k ln N + N k; A = U - T S"
        )
        ctx.formatted = f"ln Z = {fmt_sig(ln_z, 8)}, U = {fmt_sig(out['U'], 6)} J, S = {fmt_sig(out['S'], 6)} J/K."
    else:
        ctx.formatted = f"ln Z = {fmt_sig(ln_z, 8)} per molecule (give N or n_mol for U and S)."
    if "Z" not in out:
        ctx.warn("Z is beyond the floating-point range: use ln_Z.")
    return out
