"""eng_calc, modern physics and special relativity (D49).

CODATA 2022 constants (`constants.py`). Exact relativistic formulas: no classical
approximation, which stays in the result only for comparison (`*_classical`).
D116, mode `cherenkov` (page "Cherenkov radiation", section "Emission angle"): cos theta = 1 / (n beta),
emission only for beta > 1 / n; threshold gamma = 1 / sqrt(1 - 1/n^2) and kinetic energy (gamma - 1) m c^2.
"""

from __future__ import annotations

import math
from typing import Any, cast

from scipy.integrate import quad

from ..constants import lookup
from ..envelope import Ctx
from ..errors import ToolError
from .eng_base import choice, fmt_sig, integer, need, one_of, only, op, qty, temperature, units_for
from .eng_nuc import KEV_J, nuclear_mass_kev, nuclide_key, nuclide_za

C = 299792458.0
E_CH = 1.602176634e-19
H = lookup("Planck constant")[1]
HBAR = lookup("reduced Planck constant")[1]
M_E = lookup("electron mass")[1]
MU_B = lookup("Bohr magneton")[1]
G_E = lookup("electron g factor")[1]
RY_EV = lookup("Rydberg constant times hc in eV")[1]
A0 = lookup("Bohr radius")[1]
ALPHA = lookup("fine-structure constant")[1]
SIGMA = lookup("Stefan-Boltzmann constant")[1]
B_WIEN = lookup("Wien wavelength displacement law constant")[1]
B_WIEN_F = lookup("Wien frequency displacement law constant")[1]
C1L = lookup("first radiation constant for spectral radiance")[1]
C2 = lookup("second radiation constant")[1]
MEV_J = 1e6 * E_CH

_PARTICLES: dict[str, tuple[str, int]] = {}
for _names, _val in (
    (("electron", "e-"), ("electron mass", -1)),
    (("positron", "e+"), ("electron mass", 1)),
    (("proton", "p"), ("proton mass", 1)),
    (("neutron", "n"), ("neutron mass", 0)),
    (("alpha", "α"), ("alpha particle mass", 2)),
    (("deuteron", "d"), ("deuteron mass", 1)),
    (("muon", "mu-"), ("muon mass", -1)),
):
    for _n in _names:
        _PARTICLES[_n] = _val


def _particle(
    p: dict[str, Any], *, need_charge: bool = False
) -> tuple[float, float | None, str | None]:
    """(mass kg, charge C or None, name) from `particle` or from `m` (with optional `charge`)."""
    if p.get("particle") is not None and p.get("m") is not None:
        raise ToolError("INVALID_ARGUMENT", "Give params.particle or params.m, not both.")
    if p.get("particle") is not None:
        name = p["particle"]
        if not isinstance(name, str) or name not in _PARTICLES:
            raise ToolError(
                "INVALID_ARGUMENT",
                "params.particle: electron, proton, neutron, alpha, deuteron, muon, positron.",
            )
        const, z = _PARTICLES[name]
        return lookup(const)[1], z * E_CH, name
    if p.get("m") is None:
        raise ToolError(
            "INVALID_ARGUMENT", "The particle is needed: params.particle or the mass params.m."
        )
    m = need(p, "m", "mass")
    q = qty(p, "charge", "charge", need=need_charge, positive=False)
    return m, q, None


def _mev(x_j: float) -> float:
    return x_j / MEV_J


# ------------------------------------------------------------------ modern_physics

_MP = {
    "photon": {"wavelength", "frequency", "energy"},
    "photoelectric": {
        "work_function",
        "wavelength",
        "frequency",
        "photon_energy",
        "stopping_voltage",
    },
    "compton": {"wavelength", "energy", "angle_deg", "particle"},
    "de_broglie": {"particle", "m", "charge", "v", "kinetic_energy", "p", "voltage"},
    "hydrogen_like": {"Z", "n", "n1", "n2", "wavelength", "nuclide"},
    "blackbody": {"T", "wavelength", "wavelength1", "wavelength2", "emissivity", "area"},
    "uncertainty": {"dx", "dp", "dE", "dt", "particle", "m"},
    "angular_momentum": {"l", "s", "n", "j", "B"},
    "cherenkov": {"n", "beta", "v", "kinetic_energy", "p", "particle", "m"},
}
_SERIES = {1: "Lyman", 2: "Balmer", 3: "Paschen", 4: "Brackett", 5: "Pfund", 6: "Humphreys"}


def _photon_energy(p: dict[str, Any], keys: tuple[str, str, str], what: str) -> float:
    k = one_of(p, keys, what)
    if k == keys[0]:
        return H * C / need(p, k, "length")
    if k == keys[1]:
        return H * need(p, k, "frequency")
    return need(p, k, "energy")


@op(
    "modern_physics",
    (
        "mode",
        "wavelength",
        "frequency",
        "energy",
        "work_function",
        "photon_energy",
        "stopping_voltage",
        "angle_deg",
        "particle",
        "m",
        "charge",
        "v",
        "kinetic_energy",
        "p",
        "voltage",
        "Z",
        "n",
        "n1",
        "n2",
        "nuclide",
        "T",
        "wavelength1",
        "wavelength2",
        "emissivity",
        "area",
        "dx",
        "dp",
        "dE",
        "dt",
        "l",
        "s",
        "j",
        "B",
        "beta",
    ),
)
def _modern_physics(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    mode = choice(p, "mode", tuple(_MP))
    only(p, mode, _MP[mode])
    out = {
        "photon": _photon,
        "photoelectric": _photoelectric,
        "compton": _compton,
        "de_broglie": _de_broglie,
        "hydrogen_like": _hydrogen,
        "blackbody": _blackbody,
        "uncertainty": _uncertainty,
        "angular_momentum": _angular_momentum,
        "cherenkov": _cherenkov,
    }[mode](p, ctx)
    ctx.units = units_for(out, _UNITS)
    return out


_UNITS = {
    "theta_deg": "deg",
    "theta_max_deg": "deg",
    "beta_threshold": "1",
    "v_threshold": "m/s",
    "gamma_threshold": "1",
    "kinetic_threshold_eV": "eV",
    "L": "J*s",
    "S": "J*s",
    "J": "J*s",
    "L_z": "J*s",
    "S_z": "J*s",
    "m_l": "1",
    "m_s": "1",
    "m_j": "1",
    "j_values": "1",
    "l_values": "1",
    "g_J": "1",
    "mu_L": "J/T",
    "mu_S": "J/T",
    "mu_l_z": "J/T",
    "mu_s_z": "J/T",
    "U_l": "J",
    "U_s": "J",
    "zeeman": "J",
    "zeeman_eV": "eV",
    "wavelength": "m",
    "frequency": "Hz",
    "energy_J": "J",
    "energy_eV": "eV",
    "momentum": "kg*m/s",
    "wavenumber": "1/m",
    "photon_energy_eV": "eV",
    "work_function_eV": "eV",
    "K_max_eV": "eV",
    "stopping_voltage": "V",
    "f_cutoff": "Hz",
    "wavelength_cutoff": "m",
    "wavelength_scattered": "m",
    "shift": "m",
    "max_shift": "m",
    "compton_wavelength": "m",
    "energy_scattered_eV": "eV",
    "electron_kinetic_eV": "eV",
    "recoil_angle_deg": "deg",
    "p": "kg*m/s",
    "p_MeV_c": "MeV/c",
    "v": "m/s",
    "beta": "1",
    "gamma": "1",
    "kinetic_energy_eV": "eV",
    "wavelength_classical": "m",
    "E_n_eV": "eV",
    "E_lower_eV": "eV",
    "E_upper_eV": "eV",
    "delta_E_eV": "eV",
    "r_n": "m",
    "v_n": "m/s",
    "ionization_energy_eV": "eV",
    "reduced_mass_factor": "1",
    "n2_exact": "1",
    "ejected_kinetic_eV": "eV",
    "wavelength_max": "m",
    "frequency_max": "Hz",
    "exitance": "W/m2",
    "power": "W",
    "spectral_radiance": "W/(m2*sr*m)",
    "spectral_exitance": "W/(m2*m)",
    "band_fraction": "1",
    "band_exitance": "W/m2",
    "dp_min": "kg*m/s",
    "dx_min": "m",
    "dv_min": "m/s",
    "dt_min": "s",
    "dE_min_J": "J",
    "dE_min_eV": "eV",
}


def _photon(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    e = _photon_energy(
        p, ("wavelength", "frequency", "energy"), "the photon (wavelength, frequency, energy)"
    )
    lam = H * C / e
    ctx.formula = "E = h f = h c / lambda; p = E / c"
    ctx.formatted = f"Photon: lambda = {fmt_sig(lam)} m, E = {fmt_sig(e / E_CH)} eV."
    return {
        "wavelength": lam,
        "frequency": e / H,
        "energy_J": e,
        "energy_eV": e / E_CH,
        "momentum": e / C,
        "wavenumber": 1 / lam,
    }


def _photoelectric(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    light = [k for k in ("wavelength", "frequency", "photon_energy") if p.get(k) is not None]
    e = (
        _photon_energy(p, ("wavelength", "frequency", "photon_energy"), "the light")
        if light
        else None
    )
    if (p.get("work_function") is None) == (p.get("stopping_voltage") is None):
        raise ToolError(
            "INVALID_ARGUMENT",
            "Give the work function work_function or, with the light, the stopping potential.",
        )
    out: dict[str, Any] = {}
    if p.get("work_function") is not None:
        phi = need(p, "work_function", "energy")
    else:
        if e is None:
            raise ToolError(
                "INVALID_ARGUMENT",
                "With stopping_voltage the light is needed (wavelength, frequency or photon_energy).",
            )
        phi = e - E_CH * need(p, "stopping_voltage", "voltage", positive=False)
        if phi <= 0:
            raise ToolError("INVALID_ARGUMENT", "The stopping potential exceeds the photon energy.")
    out |= {"work_function_eV": phi / E_CH, "f_cutoff": phi / H, "wavelength_cutoff": H * C / phi}
    if e is not None:
        k = e - phi
        out |= {
            "photon_energy_eV": e / E_CH,
            "emits": k > 0,
            "K_max_eV": max(k, 0.0) / E_CH,
            "stopping_voltage": max(k, 0.0) / E_CH,
        }
        if k <= 0:
            ctx.warn("The photon has less energy than the work function: no electron emitted.")
    ctx.formula = (
        "K_max = h f - phi; V_stop = K_max / e; f_threshold = phi / h; lambda_threshold = h c / phi"
    )
    ctx.formatted = f"Work function {fmt_sig(phi / E_CH)} eV" + (
        f", K_max = {fmt_sig(out['K_max_eV'])} eV." if e is not None else "."
    )
    return out


def _compton(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    k = one_of(p, ("wavelength", "energy"), "the incident photon (wavelength or energy)")
    e = H * C / need(p, k, "length") if k == "wavelength" else need(p, k, "energy")
    theta = math.radians(need(p, "angle_deg", None, positive=False))
    if not 0 <= theta <= math.pi:
        raise ToolError(
            "INVALID_ARGUMENT", "params.angle_deg: scattering angle between 0 and 180 degrees."
        )
    m = _particle(p)[0] if p.get("particle") is not None else M_E
    lam_c = H / (m * C)
    lam = H * C / e
    # = lam_c (1 - cos theta), without cancellation at small angles (D76)
    shift = lam_c * 2 * math.sin(theta / 2) ** 2
    e2 = H * C / (lam + shift)
    phi = (
        math.degrees(math.atan2(1.0, (1 + e / (m * C**2)) * math.tan(theta / 2)))
        if theta > 0
        else None
    )
    ctx.formula = (
        "lambda' - lambda = h / (m c) (1 - cos theta); E' = h c / lambda'; "
        "cot phi = (1 + E / (m c^2)) tan(theta / 2)"
    )
    ctx.formatted = f"Shift {fmt_sig(shift)} m, scattered photon {fmt_sig(e2 / E_CH)} eV."
    return {
        "wavelength": lam,
        "wavelength_scattered": lam + shift,
        "shift": shift,
        "max_shift": 2 * lam_c,
        "compton_wavelength": lam_c,
        "energy_eV": e / E_CH,
        "energy_scattered_eV": e2 / E_CH,
        # = (E - E') / e without cancellation at small angles (D78)
        "electron_kinetic_eV": e * shift / (lam + shift) / E_CH,
        "recoil_angle_deg": phi,
    }


def _de_broglie(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    m, q, _ = _particle(p, need_charge=p.get("voltage") is not None)
    mc2 = m * C**2
    k = one_of(
        p, ("v", "kinetic_energy", "p", "voltage"), "the motion (v, kinetic_energy, p or voltage)"
    )
    if k == "v":
        beta = need(p, "v", "velocity") / C
        if beta >= 1:
            raise ToolError("INVALID_ARGUMENT", "params.v must be less than the speed of light.")
        gamma = 1 / math.sqrt((1 - beta) * (1 + beta))
        kin = (
            gamma * beta * (gamma * beta) / (gamma + 1) * mc2
        )  # = (gamma - 1) m c^2, no cancellation (D78)
        mom = gamma * m * beta * C
    elif k == "p":
        mom = need(p, "p", "momentum")
        etot = math.hypot(mom * C, mc2)
        kin = mom * C * (mom * C / (etot + mc2))  # = etot - m c^2 without cancellation (D78)
        gamma, beta = etot / mc2, mom * C / etot
    else:
        if k == "kinetic_energy":
            kin = need(p, "kinetic_energy", "energy")
        else:
            if not q:
                raise ToolError(
                    "INVALID_ARGUMENT", "With voltage a charged particle (or charge) is needed."
                )
            kin = abs(q) * need(p, "voltage", "voltage")
        etot = kin + mc2
        mom = math.sqrt(kin * (kin + 2 * mc2)) / C
        gamma, beta = etot / mc2, mom * C / etot
    lam = H / mom
    ctx.formula = (
        "lambda = h / p; p = sqrt(K (K + 2 m c^2)) / c (relativistic); classical h / sqrt(2 m K)"
    )
    ctx.formatted = (
        f"lambda = {fmt_sig(lam)} m (K = {fmt_sig(kin / E_CH)} eV, beta = {fmt_sig(beta)})."
    )
    return {
        "wavelength": lam,
        "wavelength_classical": H / math.sqrt(2 * m * kin) if kin > 0 else None,
        "p": mom,
        "p_MeV_c": mom * C / MEV_J,
        "v": beta * C,
        "beta": beta,
        "gamma": gamma,
        "kinetic_energy_eV": kin / E_CH,
    }


def _hydrogen(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    z = integer(p, "Z", 1) if p.get("Z") is not None else None
    f = 1.0
    if p.get("nuclide") is not None:
        key = nuclide_key(p["nuclide"])
        zn, _ = nuclide_za(key)
        if z is not None and z != zn:
            raise ToolError("INVALID_ARGUMENT", f"params.Z = {z} but {key} has Z = {zn}.")
        z = zn
        if z < 1:
            raise ToolError("INVALID_ARGUMENT", "The nucleus must have Z >= 1.")
        m_nuc = nuclear_mass_kev(key) * KEV_J / C**2
        f = 1 / (1 + M_E / m_nuc)
    z = z or 1
    ry = RY_EV * z * z * f

    def level(n: int) -> float:
        return -ry / n**2

    out: dict[str, Any] = {"Z": z, "reduced_mass_factor": f}
    has = {k for k in ("n", "n1", "n2", "wavelength") if p.get(k) is not None}
    if has == {"n"}:
        n = integer(p, "n", 1)
        out |= {
            "E_n_eV": level(n),
            "ionization_energy_eV": -level(n),
            "r_n": A0 * n * n / (z * f),
            "v_n": ALPHA * C * z / n,
        }
        ctx.formula = "E_n = -Ry Z^2 (mu/m_e) / n^2; r_n = a0 n^2 / (Z mu/m_e); v_n = alpha c Z / n"
        ctx.formatted = f"E_{n} = {fmt_sig(level(n), 7)} eV."
    elif has == {"n1", "n2"}:
        a, b = sorted((integer(p, "n1", 1), integer(p, "n2", 1)))
        if a == b:
            raise ToolError("INVALID_ARGUMENT", "n1 and n2 must be different.")
        de = level(b) - level(a)
        lam = H * C / (de * E_CH)
        out |= {
            "n_lower": a,
            "n_upper": b,
            "E_lower_eV": level(a),
            "E_upper_eV": level(b),
            "delta_E_eV": de,
            "wavelength": lam,
            "frequency": C / lam,
            "series": _SERIES.get(a),
        }
        ctx.formula = "1/lambda = R Z^2 (mu/m_e) (1/n1^2 - 1/n2^2); Delta E = Ry Z^2 (mu/m_e) (1/n1^2 - 1/n2^2)"
        ctx.formatted = f"Transition {b} -> {a}: lambda = {fmt_sig(lam, 7)} m (in vacuum)."
    elif has == {"n1", "wavelength"}:
        n1 = integer(p, "n1", 1)
        e_ph = H * C / need(p, "wavelength", "length") / E_CH
        inv = 1 / n1**2 - e_ph / ry
        out["photon_energy_eV"] = e_ph
        if inv <= 0:
            out |= {"ionizes": True, "ejected_kinetic_eV": e_ph + level(n1)}
            ctx.formatted = "The photon ionises the atom."
        else:
            nx = 1 / math.sqrt(inv)
            n2 = round(nx)
            out |= {"ionizes": False, "n2_exact": nx, "n2": n2, "E_n2_eV": level(n2)}
            if abs(nx - n2) > 0.05:
                ctx.warn(
                    f"n2 = {nx:.4f} is not close to an integer: the photon does not match a line."
                )
            ctx.formatted = f"Absorption from n = {n1} to n = {n2} (computed {nx:.4f})."
        ctx.formula = "1/n2^2 = 1/n1^2 - E_photon / (Ry Z^2 mu/m_e)"
    elif not has:
        out |= {"ionization_energy_eV": ry, "E_n_eV": -ry}
        ctx.formula = "E_ionisation = Ry Z^2 (mu/m_e) from the ground state"
        ctx.formatted = f"Ionisation energy {fmt_sig(ry, 7)} eV."
    else:
        raise ToolError(
            "INVALID_ARGUMENT",
            "hydrogen_like: give n (one level), or n1 and n2 (a transition), or n1 and wavelength.",
        )
    if f == 1.0:
        ctx.warn(
            "Nucleus of infinite mass (Ry = 13.6057 eV); with params.nuclide the reduced mass is used."
        )
    return out


def _planck_x(x: float) -> float:
    return x**3 * math.exp(-x) / -math.expm1(-x) if x > 0 else 0.0


def _blackbody(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    t = temperature(p, "T")
    assert t is not None
    eps = qty(p, "emissivity", None, default=1.0) or 1.0
    if eps > 1:
        raise ToolError("INVALID_ARGUMENT", "params.emissivity is between 0 and 1.")
    m_tot = eps * SIGMA * t**4
    out: dict[str, Any] = {
        "wavelength_max": B_WIEN / t,
        "frequency_max": B_WIEN_F * t,
        "exitance": m_tot,
    }
    formulas = ["lambda_max = b / T; f_max = b' T; M = eps sigma T^4"]
    if p.get("area") is not None:
        out["power"] = m_tot * need(p, "area", "area")
    if p.get("wavelength") is not None:
        lam = need(p, "wavelength", "length")
        x = C2 / (lam * t)
        # D75: far in the Wien tail exp(x) overflows; there 1 / (e^x - 1) = e^-x within e^-700, and the logarithm
        # keeps the product out of the subnormal range
        bl = C1L / lam**5 / math.expm1(x) if x < 700 else math.exp(math.log(C1L / lam**5) - x)
        out |= {"spectral_radiance": eps * bl, "spectral_exitance": eps * math.pi * bl}
        formulas.append("B_lambda = c1L / lambda^5 / (exp(c2 / (lambda T)) - 1) (Planck)")
    if (p.get("wavelength1") is None) != (p.get("wavelength2") is None):
        raise ToolError("INVALID_ARGUMENT", "The band needs wavelength1 and wavelength2.")
    if p.get("wavelength1") is not None:
        l1, l2 = sorted((need(p, "wavelength1", "length"), need(p, "wavelength2", "length")))
        x_hi, x_lo = C2 / (l1 * t), C2 / (l2 * t)
        val = cast(tuple[float, float], quad(_planck_x, x_lo, x_hi, limit=200))[0]
        frac = 15 / math.pi**4 * val
        out |= {"band_fraction": frac, "band_exitance": frac * m_tot}
        formulas.append("fraction = 15/pi^4 integral x^3/(e^x - 1) dx, x = c2/(lambda T)")
    ctx.formula = "; ".join(formulas)
    ctx.formatted = f"lambda_max = {fmt_sig(B_WIEN / t)} m, M = {fmt_sig(m_tot)} W/m2."
    return out


def _uncertainty(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    k = one_of(p, ("dx", "dp", "dE", "dt"), "the known uncertainty (dx, dp, dE or dt)")
    out: dict[str, Any] = {}
    if k in ("dE", "dt"):
        if p.get("particle") is not None or p.get("m") is not None:
            raise ToolError("INVALID_ARGUMENT", "The mass is used only with dx or dp.")
        if k == "dE":
            out["dt_min"] = HBAR / (2 * need(p, "dE", "energy"))
        else:
            de = HBAR / (2 * need(p, "dt", "time"))
            out |= {"dE_min_J": de, "dE_min_eV": de / E_CH}
        ctx.formula = "Delta E Delta t >= hbar / 2"
    else:
        dp = HBAR / (2 * need(p, "dx", "length")) if k == "dx" else need(p, "dp", "momentum")
        if k == "dx":
            out["dp_min"] = dp
        else:
            out["dx_min"] = HBAR / (2 * dp)
        if p.get("particle") is not None or p.get("m") is not None:
            out["dv_min"] = dp / _particle(p)[0]
        ctx.formula = "Delta x Delta p >= hbar / 2"
    ctx.formatted = ", ".join(f"{a} = {fmt_sig(b)}" for a, b in out.items()) + "."
    return out


# ------------------------------------------------------------------ angular momentum (D99)

_AM_MAX = 1000  # largest l and s: the lists have 2 l + 1 entries


def _half_integer(p: dict[str, Any], key: str, default: float | None = None) -> float:
    raw = p.get(key, default)
    if raw is None:
        raise ToolError("INVALID_ARGUMENT", f"Missing params.{key} (integer or half-integer).")
    if isinstance(raw, bool) or not isinstance(raw, int | float) or not math.isfinite(raw):
        raise ToolError("INVALID_ARGUMENT", f"params.{key} must be a number.")
    if raw < 0 or not float(2 * raw).is_integer():
        raise ToolError(
            "INVALID_ARGUMENT", f"params.{key} must be an integer or a half-integer >= 0."
        )
    if raw > _AM_MAX:
        raise ToolError("LIMIT_EXCEEDED", f"params.{key} is at most {_AM_MAX}.")
    return float(raw)


def _ladder(q: float) -> list[float]:
    """-q, -q + 1, ..., q."""
    return [-q + k for k in range(round(2 * q) + 1)]


def _angular_momentum(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    ell = integer(p, "l", 0)
    if ell > _AM_MAX:
        raise ToolError("LIMIT_EXCEEDED", f"params.l is at most {_AM_MAX}.")
    s = _half_integer(p, "s", 0.5)
    out: dict[str, Any] = {}
    if p.get("n") is not None:
        n = integer(p, "n", 1)
        if ell > n - 1:
            raise ToolError(
                "INVALID_ARGUMENT", f"With n = {n} the orbital number l is 0 ... {n - 1}."
            )
        out["l_values"] = list(range(n))
    js = [abs(ell - s) + k for k in range(round(ell + s - abs(ell - s)) + 1)]
    if p.get("j") is not None:
        j_sel = _half_integer(p, "j")
        if j_sel not in js:
            raise ToolError(
                "INVALID_ARGUMENT",
                f"params.j must be one of {', '.join(fmt_sig(x) for x in js)} (|l - s| ... l + s).",
            )
        js = [j_sel]
    g_s = abs(G_E)
    g_j: list[float | None] = []
    for j in js:
        if j == 0:
            g_j.append(None)  # no magnetic moment and no Zeeman shift
            continue
        jj, ss, ll = j * (j + 1), s * (s + 1), ell * (ell + 1)
        g_j.append((jj - ss + ll) / (2 * jj) + g_s * (jj + ss - ll) / (2 * jj))
    m_l, m_s = list(range(-ell, ell + 1)), _ladder(s)
    out |= {
        "L": HBAR * math.sqrt(ell * (ell + 1)),
        "S": HBAR * math.sqrt(s * (s + 1)),
        "m_l": m_l,
        "m_s": m_s,
        "L_z": [m * HBAR for m in m_l],
        "S_z": [m * HBAR for m in m_s],
        "j_values": js,
        "J": [HBAR * math.sqrt(j * (j + 1)) for j in js],
        "m_j": [_ladder(j) for j in js],
        "g_J": g_j,
        "mu_L": MU_B * math.sqrt(ell * (ell + 1)),
        "mu_l_z": [-m * MU_B for m in m_l],
        "mu_S": g_s * MU_B * math.sqrt(s * (s + 1)),
        "mu_s_z": [G_E * m * MU_B for m in m_s],
    }
    ctx.formula = (
        "|L| = hbar sqrt(l(l+1)), |S| = hbar sqrt(s(s+1)), |J| = hbar sqrt(j(j+1)), j = |l - s| ... l + s; "
        "L_z = m_l hbar, S_z = m_s hbar; mu_l,z = -m_l mu_B, mu_s,z = g_e m_s mu_B (g_e < 0); "
        "g_J = (j(j+1) - s(s+1) + l(l+1)) / (2 j(j+1)) + |g_e| (j(j+1) + s(s+1) - l(l+1)) / (2 j(j+1))"
    )
    if p.get("B") is not None:
        b = need(p, "B", "magnetic_flux_density")
        out["U_l"] = [m * MU_B * b for m in m_l]
        out["U_s"] = [-G_E * m * MU_B * b for m in m_s]
        out["zeeman"] = [
            [0.0] if g is None else [g * m * MU_B * b for m in _ladder(j)]
            for j, g in zip(js, g_j, strict=True)
        ]
        out["zeeman_eV"] = [[e / E_CH for e in row] for row in out["zeeman"]]
        ctx.formula += "; U = -mu_z B; weak-field Zeeman E = g_J m_j mu_B B"
    ctx.formatted = (
        f"l = {ell}, s = {fmt_sig(s)}: |L| = {fmt_sig(out['L'], 6)} J s, j = "
        f"{', '.join(fmt_sig(j) for j in js)}."
    )
    return out


def _cherenkov(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    """D116, page "Cherenkov radiation": cos theta = 1 / (n beta), emission only for beta > 1 / n."""
    n = need(p, "n", None)
    if n <= 1:
        raise ToolError(
            "INVALID_ARGUMENT",
            "params.n (refractive index of the medium) must be > 1: the threshold is beta = 1/n.",
        )
    k = one_of(p, ("beta", "v", "kinetic_energy", "p"), "the motion (beta, v, kinetic_energy or p)")
    has_mass = p.get("particle") is not None or p.get("m") is not None
    m = _particle(p)[0] if has_mass else None
    if k in ("kinetic_energy", "p") and m is None:
        raise ToolError("INVALID_ARGUMENT", f"With {k} the mass is needed (particle or m).")
    # gm1 = gamma - 1 kept apart, as in special_relativity (D78)
    if k in ("beta", "v"):
        beta = _beta_of(p, "v", "beta", signed=False)
        gamma = 1 / math.sqrt((1 - beta) * (1 + beta))
        gm1 = gamma * beta * (gamma * beta) / (gamma + 1)
    else:
        assert m is not None
        mc2 = m * C**2
        if k == "kinetic_energy":
            gm1 = need(p, "kinetic_energy", "energy") / mc2
        else:
            q = need(p, "p", "momentum") * C / mc2
            gm1 = q * (q / (math.hypot(q, 1.0) + 1))
        gamma = 1 + gm1
        beta = math.sqrt(gm1) * math.sqrt(gm1 + 2) / gamma
    g_t = n / math.sqrt((n - 1) * (n + 1))  # 1 / sqrt(1 - 1/n^2)
    emits = n * beta > 1
    out: dict[str, Any] = {
        "beta": beta,
        "v": beta * C,
        "gamma": gamma,
        "emits": emits,
        "theta_deg": math.degrees(math.acos(1 / (n * beta))) if emits else None,
        "theta_max_deg": math.degrees(math.acos(1 / n)),
        "beta_threshold": 1 / n,
        "v_threshold": C / n,
        "gamma_threshold": g_t,
    }
    ctx.formula = (
        "cos theta = 1 / (n beta), emission for beta > 1/n; v_threshold = c / n; "
        "gamma_threshold = 1 / sqrt(1 - 1/n^2)"
    )
    if m is not None:
        mc2 = m * C**2
        out["kinetic_energy_eV"] = gm1 * mc2 / E_CH
        # gamma_t - 1 = (gamma_t^2 - 1) / (gamma_t + 1) with gamma_t^2 - 1 = 1 / (n^2 - 1)
        out["kinetic_threshold_eV"] = mc2 / ((n - 1) * (n + 1) * (g_t + 1)) / E_CH
        ctx.formula += "; K_threshold = (gamma_threshold - 1) m c^2"
    if emits:
        ctx.formatted = f"Cherenkov angle {fmt_sig(out['theta_deg'], 6)} deg (threshold beta = {fmt_sig(1 / n, 6)})."
    else:
        ctx.warn(
            f"beta = {fmt_sig(beta, 6)} is not above the threshold 1/n = {fmt_sig(1 / n, 6)}: "
            "no Cherenkov radiation."
        )
        ctx.formatted = f"No Cherenkov radiation: beta = {fmt_sig(beta, 6)} <= 1/n."
    return out


# ------------------------------------------------------------------ special_relativity

_SR = {
    "kinematics": {
        "v",
        "beta",
        "gamma",
        "kinetic_energy",
        "p",
        "total_energy",
        "m",
        "particle",
        "proper_time",
        "proper_length",
    },
    "velocity_addition": {"u", "u_beta", "v", "v_beta"},
    "doppler": {
        "f",
        "wavelength",
        "beta",
        "v",
        "z",
        "f_observed",
        "wavelength_observed",
        "angle_deg",
        "angle_frame",
    },
    "lorentz": {"x", "t", "v", "beta"},
    "mass_energy": {"m", "particle", "energy"},
}


def _beta_of(
    p: dict[str, Any], vkey: str, bkey: str, *, signed: bool, light: bool = False
) -> float:
    k = one_of(p, (vkey, bkey), f"the velocity ({vkey} in m/s or {bkey} = v/c)")
    b = (
        need(p, vkey, "velocity", positive=not signed) / C
        if k == vkey
        else need(p, bkey, None, positive=not signed)
    )
    if abs(b) > 1 or (abs(b) == 1 and not light):
        bound = "must not exceed" if light else "must be less than"  # D108: light may move at c
        raise ToolError("INVALID_ARGUMENT", f"|{k}| {bound} the speed of light.")
    return b


@op(
    "special_relativity",
    (
        "mode",
        "v",
        "beta",
        "gamma",
        "kinetic_energy",
        "p",
        "total_energy",
        "m",
        "particle",
        "proper_time",
        "proper_length",
        "u",
        "u_beta",
        "v_beta",
        "f",
        "wavelength",
        "z",
        "f_observed",
        "wavelength_observed",
        "angle_deg",
        "angle_frame",
        "x",
        "t",
        "energy",
    ),
)
def _special_relativity(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    mode = choice(p, "mode", tuple(_SR))
    only(p, mode, _SR[mode])
    out = {
        "kinematics": _kinematics,
        "velocity_addition": _vel_add,
        "doppler": _doppler,
        "lorentz": _lorentz,
        "mass_energy": _mass_energy,
    }[mode](p, ctx)
    ctx.units = units_for(out, _SR_UNITS)
    return out


_SR_UNITS = {
    "beta": "1",
    "v": "m/s",
    "gamma": "1",
    "E0_J": "J",
    "E0_MeV": "MeV",
    "E_J": "J",
    "E_MeV": "MeV",
    "K_J": "J",
    "K_MeV": "MeV",
    "K_classical_J": "J",
    "p": "kg*m/s",
    "p_MeV_c": "MeV/c",
    "time": "s",
    "length": "m",
    "beta_classical": "1",
    "f_observed": "Hz",
    "wavelength_observed": "m",
    "z": "1",
    "angle_receiver_deg": "deg",
    "angle_source_deg": "deg",
    "x_prime": "m",
    "t_prime": "s",
    "interval_s2": "m2",
    "energy_J": "J",
    "energy_MeV": "MeV",
    "energy_kWh": "kWh",
    "m": "kg",
    "m_u": "u",
}


def _kinematics(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    keys = ("v", "beta", "gamma", "kinetic_energy", "p", "total_energy")
    k = one_of(
        p, keys, "the velocity or the energy (v, beta, gamma, kinetic_energy, p, total_energy)"
    )
    has_mass = p.get("particle") is not None or p.get("m") is not None
    m = _particle(p)[0] if has_mass else None
    if k in ("kinetic_energy", "p", "total_energy") and m is None:
        raise ToolError("INVALID_ARGUMENT", f"With {k} the mass is needed (particle or m).")
    # gm1 = gamma - 1, kept apart so that it does not cancel at low speed (D78)
    if k in ("v", "beta"):
        beta = _beta_of(p, "v", "beta", signed=False)
        gamma = 1 / math.sqrt((1 - beta) * (1 + beta))
        gm1 = gamma * beta * (gamma * beta) / (gamma + 1)  # gamma^2 - 1 = (gamma beta)^2
    elif k == "gamma":
        gamma = need(p, "gamma", None)
        if gamma < 1:
            raise ToolError("INVALID_ARGUMENT", "params.gamma must be >= 1.")
        gm1 = gamma - 1
        beta = math.sqrt(gm1) * math.sqrt(gm1 + 2) / gamma
    else:
        assert m is not None
        e0 = m * C**2
        if k == "kinetic_energy":
            gm1 = need(p, "kinetic_energy", "energy") / e0
        elif k == "total_energy":
            etot = need(p, "total_energy", "energy")
            if etot < e0:
                raise ToolError(
                    "INVALID_ARGUMENT", "The total energy is less than the rest energy."
                )
            gm1 = (etot - e0) / e0
        else:
            q = need(p, "p", "momentum") * C / e0
            gm1 = q * (q / (math.hypot(q, 1.0) + 1))  # gamma^2 - 1 = (p c / (m c^2))^2
        gamma = 1 + gm1
        beta = math.sqrt(gm1) * math.sqrt(gm1 + 2) / gamma
    out: dict[str, Any] = {"beta": beta, "v": beta * C, "gamma": gamma}
    formulas = ["gamma = 1 / sqrt(1 - beta^2)"]
    if m is not None:
        e0 = m * C**2
        pm = gamma * m * beta * C
        out |= {
            "E0_J": e0,
            "E0_MeV": _mev(e0),
            "E_J": gamma * e0,
            "E_MeV": _mev(gamma * e0),
            "K_J": gm1 * e0,
            "K_MeV": _mev(gm1 * e0),
            "K_classical_J": 0.5 * m * (beta * C) ** 2,
            "p": pm,
            "p_MeV_c": pm * C / MEV_J,
        }
        formulas.append("E = gamma m c^2; K = (gamma - 1) m c^2; p = gamma m v")
    if p.get("proper_time") is not None:
        out["time"] = gamma * need(p, "proper_time", "time")
        formulas.append("Delta t = gamma Delta tau")
    if p.get("proper_length") is not None:
        out["length"] = need(p, "proper_length", "length") / gamma
        formulas.append("L = L0 / gamma")
    ctx.formula = "; ".join(formulas)
    ctx.formatted = f"beta = {fmt_sig(beta, 7)}, gamma = {fmt_sig(gamma, 7)}."
    return out


def _vel_add(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    bu = _beta_of(p, "u", "u_beta", signed=True, light=True)
    bv = _beta_of(p, "v", "v_beta", signed=True)
    b = (bu + bv) / (1 + bu * bv)
    ctx.formula = "u = (u' + v) / (1 + u' v / c^2)"
    ctx.formatted = f"u = {fmt_sig(b, 7)} c."
    return {"beta": b, "v": b * C, "beta_classical": bu + bv}


def _cos_deg(deg: float) -> float:
    """cos of an angle in degrees, exactly 0 at 90 (math.cos(math.radians(90)) = 6.1e-17 moved z of the
    transverse case at beta = 1e-9 by 1.2e-7 of its value)."""
    return 0.0 if deg == 90 else math.cos(math.radians(deg))


def _acos_deg(c: float) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _doppler_angle(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    """D119: page "Relativistic Doppler effect", section "Motion in an arbitrary direction". theta = 0 along the
    line of sight and beta > 0 moving away, as without the angle; the angle is given in the frame of the receiver
    or of the source, the other one comes from the relativistic aberration."""
    example = (
        '{"mode": "doppler", "f": 5e14, "beta": 0.3, "angle_deg": 90, "angle_frame": "receiver"}'
    )
    if p.get("angle_deg") is None:
        raise ToolError(
            "INVALID_ARGUMENT",
            "params.angle_frame needs params.angle_deg, the angle in degrees.",
            example,
        )
    frame = choice(p, "angle_frame", ("receiver", "source"))
    deg = need(p, "angle_deg", None, positive=False)
    if not 0 <= deg <= 180:
        raise ToolError(
            "INVALID_ARGUMENT", "params.angle_deg: an angle between 0 and 180 degrees.", example
        )
    beta = _beta_of(p, "v", "beta", signed=True)
    g = 1 / math.sqrt((1 - beta) * (1 + beta))
    c = _cos_deg(deg)
    if frame == "receiver":
        c_r, c_s = c, (c + beta) / (1 + beta * c)
    else:  # aberration: cos theta_r = (cos theta_s - beta) / (1 - beta cos theta_s)
        c_s, c_r = c, (c - beta) / (1 - beta * c)
    # z = f_s / f_r - 1 = gamma (1 + beta cos theta_r) - 1, with gamma - 1 = beta^2 gamma^2 / (gamma + 1): no
    # cancellation at small beta (as D78)
    out: dict[str, Any] = {
        "beta": beta,
        "gamma": g,
        "z": beta * beta * g * g / (g + 1) + g * beta * c_r,
    }
    out["angle_receiver_deg"] = deg if frame == "receiver" else _acos_deg(c_r)
    out["angle_source_deg"] = deg if frame == "source" else _acos_deg(c_s)
    k = one_of(p, ("f", "wavelength"), "the source (f or wavelength)")
    # f_s / f_r = gamma (1 + beta cos theta_r), or f_r = gamma (1 - beta cos theta_s) f_s (Einstein 1905)
    ratio = g * (1 + beta * c_r) if frame == "receiver" else 1 / (g * (1 - beta * c_s))
    if k == "f":
        out["f_observed"] = need(p, "f", "frequency") / ratio
    else:
        out["wavelength_observed"] = need(p, "wavelength", "length") * ratio
    out["receding"] = beta * c_r > 0  # radial component v cos theta_r in the frame of the receiver
    ctx.formula = (
        "f_r = f_s / (gamma (1 + beta cos theta_r)) = gamma (1 - beta cos theta_s) f_s; "
        "cos theta_r = (cos theta_s - beta) / (1 - beta cos theta_s)"
    )
    ctx.formatted = (
        f"z = {fmt_sig(out['z'], 7)}, theta_r = {fmt_sig(out['angle_receiver_deg'], 7)} deg, "
        f"theta_s = {fmt_sig(out['angle_source_deg'], 7)} deg."
    )
    return out


def _doppler(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    angle = p.get("angle_deg") is not None or p.get("angle_frame") is not None
    inverse = (
        p.get("z") is not None
        or (p.get("f") is not None and p.get("f_observed") is not None)
        or (p.get("wavelength") is not None and p.get("wavelength_observed") is not None)
    )
    if inverse and angle:
        raise ToolError(
            "INVALID_ARGUMENT",
            "The angle works with beta or v and the source (f or wavelength), not with z or the observed values.",
        )
    if angle:
        return _doppler_angle(p, ctx)
    if inverse:
        if p.get("beta") is not None or p.get("v") is not None:
            raise ToolError(
                "INVALID_ARGUMENT", "With z or with the observed values the velocity is the result."
            )
        # z = r - 1 from a difference that is exact when r is close to 1 (D78)
        if p.get("z") is not None:
            z = need(p, "z", None, positive=False)
        elif p.get("f") is not None:
            f_obs = need(p, "f_observed", "frequency")
            z = (need(p, "f", "frequency") - f_obs) / f_obs
        else:
            lam = need(p, "wavelength", "length")
            z = (need(p, "wavelength_observed", "length") - lam) / lam
        r = 1 + z
        if r <= 0:
            raise ToolError("INVALID_ARGUMENT", "z must be greater than -1.")
        beta = z * (z + 2) / (r * r + 1)  # r^2 - 1 = z (z + 2)
        out: dict[str, Any] = {"beta": beta, "v": beta * C, "z": z}
    else:
        beta = _beta_of(p, "v", "beta", signed=True)
        r = math.sqrt((1 + beta) / (1 - beta))
        # z = r - 1 = (r^2 - 1) / (r + 1) = 2 beta / ((1 - beta)(r + 1)), no cancellation at small beta (D78)
        out = {"beta": beta, "z": 2 * beta / ((1 - beta) * (r + 1))}
        k = one_of(p, ("f", "wavelength"), "the source (f or wavelength)")
        if k == "f":
            out["f_observed"] = need(p, "f", "frequency") / r
        else:
            out["wavelength_observed"] = need(p, "wavelength", "length") * r
    out["receding"] = beta > 0
    ctx.formula = (
        "lambda_obs / lambda = f / f_obs = sqrt((1 + beta) / (1 - beta)), beta > 0 when receding"
    )
    ctx.formatted = f"beta = {fmt_sig(beta, 7)}, z = {fmt_sig(out['z'], 7)}."
    return out


def _lorentz(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    x = need(p, "x", "length", positive=False)
    t = need(p, "t", "time", positive=False)
    beta = _beta_of(p, "v", "beta", signed=True)
    gamma = 1 / math.sqrt((1 - beta) * (1 + beta))
    s2 = (C * t) ** 2 - x * x
    kind = (
        "lightlike"
        if math.isclose(s2, 0.0, abs_tol=1e-9 * max((C * t) ** 2, x * x))
        else ("timelike" if s2 > 0 else "spacelike")
    )
    ctx.formula = "x' = gamma (x - v t); t' = gamma (t - v x / c^2); s^2 = (c t)^2 - x^2"
    out = {
        "gamma": gamma,
        "x_prime": gamma * (x - beta * C * t),
        "t_prime": gamma * (t - beta * x / C),
        "interval_s2": s2,
        "interval_type": kind,
    }
    ctx.formatted = f"x' = {fmt_sig(out['x_prime'])} m, t' = {fmt_sig(out['t_prime'])} s."
    return out


def _mass_energy(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    if p.get("energy") is not None:
        if p.get("m") is not None or p.get("particle") is not None:
            raise ToolError("INVALID_ARGUMENT", "Give the mass or the energy, not both.")
        e = need(p, "energy", "energy")
        m = e / C**2
    else:
        m = _particle(p)[0]
        e = m * C**2
    ctx.formula = "E = m c^2"
    ctx.formatted = f"E = {fmt_sig(e)} J = {fmt_sig(_mev(e))} MeV for m = {fmt_sig(m)} kg."
    return {
        "m": m,
        "m_u": m / lookup("atomic mass constant")[1],
        "energy_J": e,
        "energy_MeV": _mev(e),
        "energy_kWh": e / 3.6e6,
    }
