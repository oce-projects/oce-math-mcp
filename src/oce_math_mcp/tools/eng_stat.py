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

D112 (sources: "List of equations in quantum mechanics", density of states of a free particle in 3D,
N(E) = 8 sqrt(2) pi m^(3/2) E^(1/2) / h^3; "Fermi energy", E_F = hbar^2 / (2 m) (3 pi^2 n)^(2/3), metals
2 - 10 eV; "Fermi-Dirac statistics", "Bose-Einstein statistics", "Maxwell-Boltzmann statistics"):
- Density of states per volume and energy, g(E) = g_s / (4 pi^2) (2 m / hbar^2)^(3/2) sqrt(E) (the page form for
  g_s = 2); states per volume below E, (2/3) g(E) E. From the number density n: k_F = (6 pi^2 n / g_s)^(1/3),
  E_F = hbar^2 k_F^2 / (2 m), T_F = E_F / k, g(E_F) = 3 n / (2 E_F). Non-relativistic: warning above 1% of m c^2.
- Mean occupation of a state at E with x = (E - mu) / (k T): Fermi-Dirac 1 / (exp(x) + 1), Bose-Einstein
  1 / (exp(x) - 1) (only E > mu), Maxwell-Boltzmann exp(-x). Written as exp(-x) / (1 + exp(-x)), 1 / expm1(x)
  and with ln n, so there is no overflow at large |x| and no cancellation at small x.
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
from .eng_modern import _particle  # pyright: ignore[reportPrivateUsage]

K_B = lookup("Boltzmann constant")[1]
H = lookup("Planck constant")[1]
HBAR = lookup("reduced Planck constant")[1]
C_LIGHT = lookup("speed of light in vacuum")[1]
N_A = lookup("Avogadro constant")[1]
_LN2 = math.log(2)
_ROT_WARN = (
    10.0  # the classical Z_r is off by about theta_rot / (3 T): over 3% below T = 10 theta_rot
)
_EV = lookup("elementary charge")[1]
_NR_WARN = (
    0.01  # kinetic energy above 1% of m c^2: the non-relativistic g(E) is off by about E / (m c^2)
)
_LN_FLOAT = 700.0  # |ln n| above this: n is beyond the floating-point range
_MB_WARN = math.log(0.1)  # Maxwell-Boltzmann needs n << 1
_STATS = ("fermi_dirac", "bose_einstein", "maxwell_boltzmann")

_MODES: dict[str, set[str]] = {
    "entropy": {"probabilities", "omega", "ln_omega"},
    "entropy_heat": {"Q", "T", "C", "T1", "T2", "T_reservoir"},
    "maxwell_juttner": {"T", "m", "p", "gamma"},
    "partition_functions": {"T", "m", "M_molar", "V", "nu", "omega", "I", "sigma", "N", "n_mol"},
    "density_of_states": {"particle", "m", "E", "n", "V", "g_s"},
    "occupation": {"statistics", "E", "mu", "T", "g"},
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
    "g_E": "1/(J*m3)",
    "N_below": "1/m3",
    "g_E_total": "1/J",
    "N_below_total": "1",
    "g_s": "1",
    "E_F": "J",
    "E_F_eV": "eV",
    "T_F": "K",
    "k_F": "1/m",
    "g_E_F": "1/(J*m3)",
    "N_total": "1",
    "g_E_F_total": "1/J",
    "n": "1",
    "ln_n": "1",
    "x": "1",
    "n_total": "1",
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
        "density_of_states": _mode_dos,
        "occupation": _mode_occupation,
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


def _g_of_e(m: float, e: float, g_s: int) -> float:
    """Density of states per volume and energy of a free particle in 3D (non-relativistic)."""
    return g_s / (4 * math.pi**2) * (2 * m / (HBAR * HBAR)) ** 1.5 * math.sqrt(e)


def _mode_dos(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    m, _, _ = _particle(p)
    g_s = integer(p, "g_s", 1) if p.get("g_s") is not None else 2
    key = one_of(
        p, ("E", "n"), "the density of states (the energy E, or the number density n in 1/m3)"
    )
    vol = qty(p, "V", "volume", need=False)
    out: dict[str, Any] = {"g_s": g_s}
    if key == "E":
        e = need(p, "E", "energy")
        g = _g_of_e(m, e, g_s)
        out["g_E"] = g
        out["N_below"] = 2 * g * e / 3
        if vol is not None:
            out["g_E_total"] = g * vol
            out["N_below_total"] = out["N_below"] * vol
        ctx.formatted = (
            f"g(E) = {fmt_sig(g, 6)} 1/(J m3); {fmt_sig(out['N_below'], 6)} states per m3 below E."
        )
    else:
        n = need(p, "n", None)
        k_f = (6 * math.pi**2 * n / g_s) ** (1 / 3)
        e = HBAR * HBAR * k_f * k_f / (2 * m)
        g_f = _g_of_e(m, e, g_s)
        out.update({"k_F": k_f, "E_F": e, "E_F_eV": e / _EV, "T_F": e / K_B, "g_E_F": g_f})
        if vol is not None:
            out["N_total"] = n * vol
            out["g_E_F_total"] = g_f * vol
        ctx.formatted = f"E_F = {fmt_sig(e / _EV, 6)} eV, T_F = {fmt_sig(e / K_B, 6)} K."
    ctx.formula = (
        "g(E) = g_s / (4 pi^2) (2 m / hbar^2)^(3/2) sqrt(E) (for g_s = 2: 8 sqrt(2) pi m^(3/2) sqrt(E) / h^3); "
        "states below E: (2/3) g(E) E; k_F = (6 pi^2 n / g_s)^(1/3), E_F = hbar^2 k_F^2 / (2 m), T_F = E_F / k"
    )
    rest = m * C_LIGHT * C_LIGHT
    if e > _NR_WARN * rest:
        ctx.warn(
            f"The energy is {fmt_sig(100 * e / rest, 3)}% of m c^2: the non-relativistic formulas have "
            "corrections of the order of E / (m c^2)."
        )
    return out


def _occupation(stat: str, x: float) -> tuple[float | None, float]:
    """(n, ln n) at x = (E - mu) / (k T); n is None beyond the floating-point range."""
    if stat == "fermi_dirac":
        if x > 0:
            ex = math.exp(-x)
            n, ln_n = ex / (1 + ex), -x - math.log1p(ex)
        else:
            ex = math.exp(x)
            n, ln_n = 1 / (1 + ex), -math.log1p(ex)
    elif stat == "bose_einstein":
        ln_n = -x - math.log(-math.expm1(-x))
        n = 1 / math.expm1(x) if ln_n > -_LN_FLOAT else 0.0
    else:
        ln_n = -x
        n = math.exp(-x) if ln_n < _LN_FLOAT else 0.0
    return (n if abs(ln_n) < _LN_FLOAT else None), ln_n


def _mode_occupation(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    stat = choice(p, "statistics", _STATS)
    e = need(p, "E", "energy", positive=False)
    mu = need(p, "mu", "energy", positive=False)
    t = temperature(p, "T")
    assert t is not None
    g = integer(p, "g", 1) if p.get("g") is not None else None
    x = (e - mu) / (K_B * t)
    if stat == "bose_einstein" and x <= 0:
        raise ToolError(
            "INVALID_ARGUMENT",
            "Bose-Einstein needs E > mu: the occupation diverges at E = mu and is negative below.",
        )
    n, ln_n = _occupation(stat, x)
    out: dict[str, Any] = {"x": x, "ln_n": ln_n}
    ctx.formula = {
        "fermi_dirac": "n = 1 / (exp((E - mu) / (k T)) + 1) (Fermi-Dirac)",
        "bose_einstein": "n = 1 / (exp((E - mu) / (k T)) - 1) (Bose-Einstein, E > mu)",
        "maxwell_boltzmann": "n = exp(-(E - mu) / (k T)) (Maxwell-Boltzmann, the limit n << 1)",
    }[stat] + "; n_total = g n for a level of degeneracy g"
    if n is None:
        ctx.warn("n is beyond the floating-point range: use ln_n.")
        ctx.formatted = f"x = {fmt_sig(x, 6)}, ln n = {fmt_sig(ln_n, 8)}."
    else:
        out["n"] = n
        if g is not None:
            out["n_total"] = g * n
        ctx.formatted = f"x = {fmt_sig(x, 6)}, n = {fmt_sig(n, 8)}."
    if stat == "maxwell_boltzmann" and ln_n > _MB_WARN:
        ctx.warn(
            "n > 0.1: Maxwell-Boltzmann is the limit n << 1 of the quantum statistics; "
            "use fermi_dirac or bose_einstein."
        )
    return out
