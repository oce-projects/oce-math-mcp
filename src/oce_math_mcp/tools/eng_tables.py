"""eng_calc, operations with data tables (D44): fatigue, bolts, fillet welds,
materials database. Every table value has its source, cited here and in the result.

Sources (read on 24/11/2026):
- fatigue, Shigley method: users.wpi.edu/~cfurlong/me3320/lect14/Lect14.pdf (Se' = 0.5 Sut,
  maximum 700 MPa); www.uoanbar.edu.iq/eStoreImages/Bank/3249.pdf (tables 3-1 of ka and 3-3 of ke,
  kb = 1.24 d^-0.107); www.secs.oakland.edu/~latcha/ME4300/SM_PDF/CH6.pdf (eq. 6-13, 6-14, 6-15);
  highered.mheducation.com/sites/dl/free/1456267566/1110465/Stochastic_Analysis.pdf (kc);
  Goodman, Gerber, Soderberg: en.wikipedia.org/wiki/Goodman_relation; Miner:
  en.wikipedia.org/wiki/Fatigue_(material); EN 1993-1-9:
  www.ideastatica.com/support-center/fatigue-analysis-according-to-en-1993-1-9.
- bolts: ISO 262 coarse pitches from en.wikipedia.org/wiki/ISO_metric_screw_thread; tensile stress area
  ISO 898-1 (engineersedge.com/thread_strength/thread_tensile_stress_area.htm); EN 1993-1-8
  tables 3.1 and 3.4 (eurocodeapplied.com/design/en1993/bolt-design-properties,
  www.ideastatica.com/support-center/support-center-knowledge-base/check-of-bolts-and-preloaded-bolts-according-to-eurocode,
  orangebook.arcelormittal.com/node/170); rule of the class name:
  en.wikipedia.org/wiki/Screw (Mechanical classifications); torque T = K d F and VDI 2230:
  www.machinedesign.com/fastening-joining/article/21834566/understanding-the-nut-factor-in-thr.
- welds: EN 1993-1-8 4.5.3 (ideastatica check-of-welds-according-to-eurocode,
  repository.tudelft.nl/file/File_30d99fd2-af2e-43c9-bf1f-e8eeb97e0a55); beta_w table 4.1
  (help.solidworks.com/2012/English/SolidWorks/Cworks/r_Correlation_Factor_EuroCode.htm); fu for
  EN 10025-2 from EN 1993-1-1 table 3.1 (blueprints, below).
"""

from __future__ import annotations

import math
import re
from typing import Any

from ..envelope import Ctx
from ..errors import ToolError
from ..fmt import fmt_float
from .eng_base import choice, need, only, op, qty, units_for

MPA = 1e6
_EDGE = 1e-12  # relative tolerance on the slope-change points of the EN 1993-1-9 curves

# ------------------------------------------------------------------ fatigue_life

_KA = {  # a in MPa, b (Shigley, table 6-2 / uoanbar 3-1)
    "ground": (1.58, -0.085),
    "machined": (4.51, -0.265),
    "hot_rolled": (57.7, -0.718),
    "as_forged": (272.0, -0.995),
}
_KC = {"bending": 1.0, "axial": 0.85, "torsion": 0.59}
_KE = {50.0: 1.000, 90.0: 0.897, 95.0: 0.868, 99.0: 0.814, 99.9: 0.753, 99.99: 0.702}
_FAT = {
    "endurance": {"Sut", "surface", "load", "d", "kb", "reliability", "kd", "kf", "Se_prime"},
    "mean_stress": {"sigma_a", "sigma_m", "Se", "Sut", "Sy"},
    "sn_life": {"sigma_a", "Sut", "Se", "f"},
    "ec3": {"delta_sigma", "delta_sigma_C", "delta_tau", "delta_tau_C", "gamma_Ff", "gamma_Mf"},
    "miner": {"blocks"},
}


def _endurance(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    Sut = need(p, "Sut", "pressure")
    surface = choice(p, "surface", tuple(_KA))
    load = choice(p, "load", tuple(_KC), "bending")
    se_p = qty(p, "Se_prime", "pressure", need=False)
    if se_p is None:
        se_p = 0.5 * Sut if Sut <= 1400 * MPA else 700 * MPA
    a, b = _KA[surface]
    ka = a * (Sut / MPA) ** b
    kb_given = qty(p, "kb", None, need=False)
    if kb_given is not None:
        kb = kb_given
    elif load == "axial":
        kb = 1.0
    else:
        d = qty(p, "d", "length", need=False)
        if d is None:
            raise ToolError(
                "INVALID_ARGUMENT", "Bending and torsion: d (diameter) or kb is needed."
            )
        dmm = d * 1000
        if not 2.79 <= dmm <= 51:
            raise ToolError(
                "INVALID_ARGUMENT",
                "kb documented only for d between 2.79 and 51 mm: for larger diameters pass kb.",
            )
        kb = 1.24 * dmm**-0.107
    rel_raw = p.get("reliability", 50)
    if (
        isinstance(rel_raw, bool)
        or not isinstance(rel_raw, int | float)
        or float(rel_raw) not in _KE
    ):
        raise ToolError(
            "INVALID_ARGUMENT", "params.reliability (%): one of 50, 90, 95, 99, 99.9, 99.99."
        )
    ke = _KE[float(rel_raw)]
    kd = qty(p, "kd", None, default=1.0)
    kf = qty(p, "kf", None, default=1.0)
    assert kd is not None and kf is not None
    kc = _KC[load]
    Se = ka * kb * kc * kd * ke * kf * se_p
    ctx.formula = (
        "Se = ka kb kc kd ke kf Se'; Se' = 0.5 Sut (Sut <= 1400 MPa) or 700 MPa; ka = a Sut^b; "
        "kb = 1.24 d^-0.107 (d in mm); kc: 1 bending, 0.85 axial, 0.59 torsion"
    )
    ctx.formatted = f"Corrected endurance limit Se = {fmt_float(Se / MPA, 4)} MPa."
    return {"Se_prime": se_p, "ka": ka, "kb": kb, "kc": kc, "kd": kd, "ke": ke, "kf": kf, "Se": Se}


def _mean_stress(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    sa = need(p, "sigma_a", "pressure")
    sm = need(p, "sigma_m", "pressure", positive=False)
    Se, Sut = need(p, "Se", "pressure"), need(p, "Sut", "pressure")
    Sy = qty(p, "Sy", "pressure", need=False)
    out: dict[str, Any]
    if sm <= 0:
        if sm < 0:
            ctx.warn("Compressive mean stress: no benefit, n = Se / sigma_a.")
        n0 = Se / sa
        out = {"n_goodman": n0, "n_gerber": n0}
        if Sy is not None:
            out["n_soderberg"] = n0
        out["sigma_ar_goodman"] = sa
    else:
        out = {"n_goodman": 1 / (sa / Se + sm / Sut)}
        qa, qb = (sm / Sut) ** 2, sa / Se  # qa n^2 + qb n - 1 = 0
        out["n_gerber"] = 2 / (qb + math.sqrt(qb * qb + 4 * qa))
        if Sy is not None:
            out["n_soderberg"] = 1 / (sa / Se + sm / Sy)
        if sm >= Sut:
            raise ToolError(
                "INVALID_ARGUMENT", "sigma_m >= Sut: the part already fails under the mean load."
            )
        out["sigma_ar_goodman"] = sa / (1 - sm / Sut)
    ctx.formula = (
        "Goodman: sa/Se + sm/Sut = 1/n; Soderberg: sa/Se + sm/Sy = 1/n; "
        "Gerber: n sa/Se + (n sm/Sut)^2 = 1; sigma_ar = sa / (1 - sm/Sut)"
    )
    ctx.formatted = (
        f"Fatigue safety factor: Goodman {fmt_float(out['n_goodman'], 4)}, Gerber {fmt_float(out['n_gerber'], 4)}"
        + (f", Soderberg {fmt_float(out['n_soderberg'], 4)}" if "n_soderberg" in out else "")
        + "."
    )
    return out


def _sn_life(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    sa, Sut, Se = (
        need(p, "sigma_a", "pressure"),
        need(p, "Sut", "pressure"),
        need(p, "Se", "pressure"),
    )
    f = need(p, "f", None)
    if not f * Sut > Se:
        raise ToolError(
            "INVALID_ARGUMENT", "f Sut > Se is needed (the line falls from 10^3 to 10^6 cycles)."
        )
    a = (f * Sut) ** 2 / Se
    b = -math.log10(f * Sut / Se) / 3
    N = None if sa <= Se else (sa / a) ** (1 / b)
    if N is not None and N < 1e3:
        ctx.warn("N < 1000 cycles: outside the 10^3-10^6 line, low-cycle fatigue zone.")
    ctx.formula = "a = (f Sut)^2 / Se; b = -(1/3) log10(f Sut / Se); N = (sigma_a / a)^(1/b)"
    ctx.formatted = (
        "Infinite life (sigma_a <= Se)."
        if N is None
        else f"Estimated life {fmt_float(N, 4)} cycles."
    )
    return {"a": a, "b": b, "N": N}


def _ec3(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    gF = qty(p, "gamma_Ff", None, default=1.0)
    gM = qty(p, "gamma_Mf", None, default=1.0)
    assert gF is not None and gM is not None
    normal = p.get("delta_sigma") is not None or p.get("delta_sigma_C") is not None
    shear = p.get("delta_tau") is not None or p.get("delta_tau_C") is not None
    if normal == shear:
        raise ToolError(
            "INVALID_ARGUMENT",
            "ec3: give delta_sigma with delta_sigma_C or delta_tau with delta_tau_C.",
        )
    if normal:
        ds, C = need(p, "delta_sigma", "pressure"), need(p, "delta_sigma_C", "pressure")
        D = (2 / 5) ** (1 / 3) * C
        L = (5 / 100) ** (1 / 5) * D
        s = gF * ds
        sc = s * (1 + _EDGE)  # the limits returned with 15 digits fall in their own segment
        if sc >= D / gM:
            N: float | None = 2e6 * (C / gM / s) ** 3
        elif sc >= L / gM:
            N = 5e6 * (D / gM / s) ** 5
        else:
            N = None
        out: dict[str, Any] = {"N": N, "delta_sigma_D": D, "delta_sigma_L": L}
        ctx.formula = (
            "EN 1993-1-9: N = 2e6 (dsC/gMf / (gFf ds))^3 sopra dsD = (2/5)^(1/3) dsC; "
            "N = 5e6 (dsD/gMf / (gFf ds))^5 up to dsL = (5/100)^(1/5) dsD; below that unlimited life"
        )
    else:
        dt, C = need(p, "delta_tau", "pressure"), need(p, "delta_tau_C", "pressure")
        L = (2 / 100) ** (1 / 5) * C
        s = gF * dt
        N = 2e6 * (C / gM / s) ** 5 if s * (1 + _EDGE) >= L / gM else None
        out = {"N": N, "delta_tau_L": L}
        ctx.formula = (
            "EN 1993-1-9 shear: N = 2e6 (dtC/gMf / (gFf dt))^5 up to dtL = (2/100)^(1/5) dtC"
        )
    ctx.formatted = (
        "Below the cut-off limit: unlimited life."
        if N is None
        else f"N = {fmt_float(N, 4)} cycles."
    )
    return out


def _miner(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    blocks = p.get("blocks")
    if not isinstance(blocks, list) or not 1 <= len(blocks) <= 1000:
        raise ToolError("INVALID_ARGUMENT", "params.blocks: list of 1 to 1000 pairs [n_i, N_i].")
    D = 0.0
    for i, bl in enumerate(blocks):
        ok_pair = (
            isinstance(bl, list)
            and len(bl) == 2
            and all(isinstance(x, int | float) and not isinstance(x, bool) for x in bl)
        )
        if not ok_pair or bl[0] < 0 or bl[1] <= 0:
            raise ToolError("INVALID_ARGUMENT", f"params.blocks[{i}]: pair [n_i >= 0, N_i > 0].")
        D += bl[0] / bl[1]
    ctx.formula = "Palmgren-Miner: D = sum(n_i / N_i), failure at D = 1"
    ctx.formatted = f"Cumulative damage D = {fmt_float(D, 4)}."
    return {"D": D, "remaining": 1 - D, "ok": D <= 1}


@op("fatigue_life", ("mode", *sorted(set().union(*_FAT.values()))))
def _fatigue(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    mode = choice(p, "mode", tuple(_FAT))
    only(p, mode, _FAT[mode])
    fn = {
        "endurance": _endurance,
        "mean_stress": _mean_stress,
        "sn_life": _sn_life,
        "ec3": _ec3,
        "miner": _miner,
    }[mode]
    out = fn(p, ctx)
    pa = {
        "Se_prime",
        "Se",
        "sigma_ar_goodman",
        "a",
        "delta_sigma_D",
        "delta_sigma_L",
        "delta_tau_L",
    }
    ctx.units = {k: ("Pa" if k in pa else "cycles" if k == "N" else "1") for k in out if k != "ok"}
    return out


# ------------------------------------------------------------------ bolt_joint

_PITCH = {  # ISO 262, coarse pitch in mm (Wikipedia, ISO metric screw thread)
    "1": 0.25, "1.2": 0.25, "1.4": 0.3, "1.6": 0.35, "1.8": 0.35, "2": 0.4, "2.5": 0.45,
    "3": 0.5, "3.5": 0.6, "4": 0.7, "5": 0.8, "6": 1.0, "7": 1.0, "8": 1.25, "10": 1.5,
    "12": 1.75, "14": 2.0, "16": 2.0, "18": 2.5, "20": 2.5, "22": 2.5, "24": 3.0, "27": 3.0,
    "30": 3.5, "33": 3.5, "36": 4.0, "39": 4.0, "42": 4.5, "45": 4.5, "48": 5.0, "52": 5.0,
    "56": 5.5, "60": 5.5, "64": 6.0,
}  # fmt: skip
_GRADE = {  # EN 1993-1-8 table 3.1: fyb, fub in MPa; alpha_v with the thread in the shear plane
    "4.6": (240, 400, 0.6), "4.8": (320, 400, 0.5), "5.6": (300, 500, 0.6),
    "5.8": (400, 500, 0.5), "6.8": (480, 600, 0.5), "8.8": (640, 800, 0.6),
    "10.9": (900, 1000, 0.5), "12.9": (1080, 1200, 0.5),
}  # fmt: skip
_BOLT_KEYS = (
    "size", "d", "P", "grade", "Fv", "Ft", "shear_planes", "threads_in_shear", "countersunk",
    "gamma_M2", "K", "mu_thread", "mu_head", "D_km", "mu_slip", "ks", "gamma_M3",
)  # fmt: skip


def _flag(p: dict[str, Any], key: str, default: bool) -> bool:
    v = p.get(key, default)
    if not isinstance(v, bool):
        raise ToolError("INVALID_ARGUMENT", f"params.{key} must be true or false.")
    return v


@op("bolt_joint", _BOLT_KEYS)
def _bolt(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    if p.get("size") is not None:
        if p.get("d") is not None or p.get("P") is not None:
            raise ToolError("INVALID_ARGUMENT", "Give size or d with P, not both.")
        m = re.fullmatch(r"M(\d+(?:\.\d+)?)", str(p["size"]))
        if m is None or m.group(1) not in _PITCH:
            raise ToolError(
                "INVALID_ARGUMENT",
                f"params.size: ISO 262 coarse-pitch threads, from M1 to M64 ({', '.join('M' + k for k in _PITCH)}).",
            )
        d, P = float(m.group(1)) / 1000, _PITCH[m.group(1)] / 1000
    else:
        d, P = need(p, "d", "length"), need(p, "P", "length")
        if d / 2 <= P:
            raise ToolError("INVALID_ARGUMENT", "params.P (pitch) too large for the diameter d.")
    grade = str(p.get("grade"))
    if grade not in _GRADE:
        raise ToolError("INVALID_ARGUMENT", f"params.grade: one of {', '.join(_GRADE)} (as text).")
    fyb, fub, av = (_GRADE[grade][0] * MPA, _GRADE[grade][1] * MPA, _GRADE[grade][2])
    if grade == "12.9":
        ctx.warn(
            "Class 12.9 outside EN 1993-1-8: nominal fyb and fub from the rule of the name; "
            "alpha_v = 0.5 as for 10.9, a cautious choice not taken from the standard."
        )
    d2, d3 = d - 0.649519 * P, d - 1.226869 * P
    As = math.pi / 4 * ((d2 + d3) / 2) ** 2
    threads = _flag(p, "threads_in_shear", True)
    A = As if threads else math.pi * d * d / 4
    if not threads:
        av = 0.6
    planes = qty(p, "shear_planes", None, default=1.0)
    gM2 = qty(p, "gamma_M2", None, default=1.25)
    assert planes is not None and gM2 is not None
    k2 = 0.63 if _flag(p, "countersunk", False) else 0.9
    FvRd = planes * av * fub * A / gM2
    FtRd = k2 * fub * As / gM2
    Fp = 0.7 * fub * As
    out: dict[str, Any] = {
        "d": d, "P": P, "d2": d2, "d3": d3, "As": As, "A": A, "fyb": fyb, "fub": fub,
        "Fv_Rd": FvRd, "Ft_Rd": FtRd, "Fp_C": Fp,
    }  # fmt: skip
    formulas = [
        "As = pi/4 ((d2 + d3)/2)^2, d2 = d - 0.649519 P, d3 = d - 1.226869 P (ISO 898-1)",
        "Fv_Rd = n alpha_v fub A / gamma_M2; Ft_Rd = k2 fub As / gamma_M2; Fp_C = 0.7 fub As (EN 1993-1-8)",
    ]
    K = qty(p, "K", None, need=False)
    if K is not None:
        out["T_K"] = K * d * Fp
        formulas.append("T = K d Fp_C")
    fr = [p.get(k) is not None for k in ("mu_thread", "mu_head", "D_km")]
    if any(fr):
        if not all(fr):
            raise ToolError(
                "INVALID_ARGUMENT", "VDI 2230 torque: mu_thread, mu_head and D_km are needed."
            )
        muG, muK = need(p, "mu_thread", None), need(p, "mu_head", None)
        Dkm = need(p, "D_km", "length")
        out["T_vdi"] = Fp * (0.159 * P + 0.578 * d2 * muG + Dkm * muK / 2)
        formulas.append("VDI 2230: T = Fp_C (0.159 P + 0.578 d2 mu_G + D_km mu_K / 2)")
    Fv = qty(p, "Fv", "force", positive=False, need=False)
    Ft = qty(p, "Ft", "force", positive=False, need=False)
    mu = qty(p, "mu_slip", None, need=False)
    if mu is not None:
        ks = qty(p, "ks", None, default=1.0)
        gM3 = qty(p, "gamma_M3", None, default=1.25)
        assert ks is not None and gM3 is not None
        out["Fs_Rd"] = ks * planes * mu * (Fp - 0.8 * (Ft or 0.0)) / gM3
        formulas.append("Fs_Rd = ks n mu (Fp_C - 0.8 Ft) / gamma_M3")
    if Fv is not None or Ft is not None:
        us, ut = (Fv or 0.0) / FvRd, (Ft or 0.0) / FtRd
        uc = us + (Ft or 0.0) / (1.4 * FtRd)
        out |= {"util_shear": us, "util_tension": ut, "util_combined": uc}
        out["ok"] = max(us, ut, uc) <= 1
        formulas.append("Fv/Fv_Rd + Ft/(1.4 Ft_Rd) <= 1")
    ctx.formula = "; ".join(formulas)
    ctx.formatted = (
        f"M{fmt_float(d * 1000, 3)} class {grade}: As = {fmt_float(As * 1e6, 4)} mm2, "
        f"Fv_Rd = {fmt_float(FvRd / 1e3, 4)} kN, Ft_Rd = {fmt_float(FtRd / 1e3, 4)} kN, "
        f"Fp_C = {fmt_float(Fp / 1e3, 4)} kN."
    )
    ctx.units = units_for(
        out,
        {
            "d": "m", "P": "m", "d2": "m", "d3": "m", "As": "m2", "A": "m2", "fyb": "Pa",
            "fub": "Pa", "Fv_Rd": "N", "Ft_Rd": "N", "Fp_C": "N", "T_K": "Nm", "T_vdi": "Nm",
            "Fs_Rd": "N", "util_shear": "1", "util_tension": "1", "util_combined": "1",
        },
    )  # fmt: skip
    return out


# ------------------------------------------------------------------ weld_fillet

_WELD_STEEL = {  # fu (MPa) for t <= 40 and 40 < t <= 80 mm; beta_w (EN 1993-1-8 table 4.1)
    "S235": (360, 360, 0.8),
    "S275": (430, 410, 0.85),
    "S355": (490, 470, 0.9),
    "S420": (520, None, 1.0),
    "S460": (540, None, 1.0),
}


@op(
    "weld_fillet",
    (
        "a",
        "z",
        "steel",
        "t",
        "fu",
        "beta_w",
        "gamma_M2",
        "F",
        "L",
        "sigma_perp",
        "tau_perp",
        "tau_par",
    ),
)
def _weld(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    if (p.get("a") is None) == (p.get("z") is None):
        raise ToolError("INVALID_ARGUMENT", "Give the throat a or the leg z, only one.")
    a = need(p, "a", "length") if p.get("a") is not None else need(p, "z", "length") / math.sqrt(2)
    if p.get("steel") is not None:
        if p.get("fu") is not None or p.get("beta_w") is not None:
            raise ToolError("INVALID_ARGUMENT", "Give steel or fu with beta_w, not both.")
        steel = choice(p, "steel", tuple(_WELD_STEEL))
        t = qty(p, "t", "length", need=False)
        thin, thick, bw = _WELD_STEEL[steel]
        if t is None or t <= 0.040:
            fu_mpa: float | None = thin
        elif t <= 0.080:
            fu_mpa = thick
        else:
            fu_mpa = None
        if fu_mpa is None:
            raise ToolError(
                "INVALID_ARGUMENT",
                f"fu of {steel} not documented for this thickness: pass fu and beta_w.",
            )
        fu = fu_mpa * MPA
    else:
        if p.get("t") is not None:
            raise ToolError("INVALID_ARGUMENT", "params.t is used only with steel.")
        fu, bw = need(p, "fu", "pressure"), need(p, "beta_w", None)
    gM2 = qty(p, "gamma_M2", None, default=1.25)
    assert gM2 is not None
    fvw = fu / (math.sqrt(3) * bw * gM2)
    out: dict[str, Any] = {"a": a, "fu": fu, "beta_w": bw, "fvw_d": fvw, "Fw_Rd": fvw * a}
    formulas = ["fvw_d = fu / (sqrt(3) beta_w gamma_M2); Fw_Rd = fvw_d a (EN 1993-1-8 4.5.3.3)"]
    L = qty(p, "L", "length", need=False)
    if L is not None:
        out["F_Rd"] = fvw * a * L
    stresses = [p.get(k) is not None for k in ("sigma_perp", "tau_perp", "tau_par")]
    if p.get("F") is not None and any(stresses):
        raise ToolError(
            "INVALID_ARGUMENT", "Check with F (simplified) or with the stresses, only one."
        )
    if p.get("F") is not None:
        if L is None:
            raise ToolError(
                "INVALID_ARGUMENT", "With F, L (effective length of the weld) is needed."
            )
        F = need(p, "F", "force")
        out["util"] = F / (fvw * a * L)
        out["ok"] = out["util"] <= 1
    elif any(stresses):
        sp, tp, tl = (
            qty(p, k, "pressure", positive=False, default=0.0) or 0.0
            for k in ("sigma_perp", "tau_perp", "tau_par")
        )
        eq = math.sqrt(sp * sp + 3 * (tp * tp + tl * tl))
        lim_eq, lim_p = fu / (bw * gM2), 0.9 * fu / gM2
        out |= {
            "sigma_eq": eq,
            "limit_eq": lim_eq,
            "limit_perp": lim_p,
            "util_eq": eq / lim_eq,
            "util_perp": abs(sp) / lim_p,
        }
        out["ok"] = max(out["util_eq"], out["util_perp"]) <= 1
        formulas.append(
            "directional: sqrt(s_perp^2 + 3 (t_perp^2 + t_par^2)) <= fu/(beta_w gamma_M2); "
            "s_perp <= 0.9 fu/gamma_M2 (EN 1993-1-8 4.5.3.2)"
        )
    ctx.formula = "; ".join(formulas)
    ctx.formatted = f"Throat {fmt_float(a * 1000, 3)} mm: resistance {fmt_float(fvw * a / 1e3, 4)} kN per metre of weld."
    ctx.units = units_for(
        out,
        {
            "a": "m", "fu": "Pa", "beta_w": "1", "fvw_d": "Pa", "Fw_Rd": "N/m", "F_Rd": "N",
            "util": "1", "sigma_eq": "Pa", "limit_eq": "Pa", "limit_perp": "Pa", "util_eq": "1",
            "util_perp": "1",
        },
    )  # fmt: skip
    return out


# ------------------------------------------------------------------ material_lookup

_W = "https://en.wikipedia.org/wiki/"
_EC3 = "https://eurocodeapplied.com/design/en1993/steel-design-properties"
_EC3_T31 = (
    "https://blueprints.readthedocs.io/en/stable/API%20reference/codes/eurocode/"
    "en_1993_1_1_2005/chapter_3_materials/table_3_1/"
)
_OVAKO = "https://steelnavigator.ovako.com/steel-grades/c45/pdf"
_SAAR = "https://en.saarstahl.com/app/uploads/2024/03/20160401102929-0503_C45.pdf"
_SWISS = "https://swisssteel-group.com/content-media/documents/Data-Sheets/Engineering-Steel/C45E-C45R_en.pdf"
_EC2 = "https://eurocodeapplied.com/design/en1992/concrete-design-properties"
_EN338 = (
    "https://blueprints.readthedocs.io/en/stable/API%20reference/codes/eurocode/en_338_2016/"
    "chapter_5_classification_of_structural_timber/table_1/"
)
_METAL = ("E", "G", "nu", "rho", "fy", "fu", "k", "c", "alpha")
_STEEL_COMMON = {"E": 210e9, "G": 81e9, "nu": 0.3, "rho": 7850.0, "alpha": 12e-6}
_C45_COMMON = {
    "E": 210e9,
    "G": 80e9,
    "nu": 0.3,
    "rho": 7800.0,
    "alpha": 12e-6,
    "c": [460.0, 480.0],
    "k": [40.0, 45.0],
}
# steels: thickness bands (maximum t in m, fy MPa, fu MPa or [min, max])
_BANDS: dict[str, tuple[str, list[tuple[float, float, Any]]]] = {
    "S235": (_EC3_T31, [(0.040, 235, 360), (0.080, 215, 360)]),
    "S275": (_EC3_T31, [(0.040, 275, 430), (0.080, 255, 410)]),
    "S355": (_EC3_T31, [(0.040, 355, 490), (0.080, 335, 470)]),
    "C45+N": (_SAAR, [(0.016, 340, 620), (0.100, 305, 580)]),
}
_UNITS_PROP = {
    "E": "Pa", "G": "Pa", "nu": "1", "rho": "kg/m3", "fy": "Pa", "fu": "Pa", "k": "W/(m*K)",
    "c": "J/(kg*K)", "alpha": "1/K", "fck": "Pa", "fcm": "Pa", "fctm": "Pa", "Ecm": "Pa",
    "unit_weight": "N/m3", "fm_k": "Pa", "E0_mean": "Pa", "rho_mean": "kg/m3",
}  # fmt: skip


def _fixed(
    values: dict[str, Any], src: str | dict[str, str], keys: tuple[str, ...] = _METAL
) -> tuple[dict[str, Any], dict[str, str]]:
    props = {k: values.get(k) for k in keys}
    source = {
        k: (src if isinstance(src, str) else src[k]) for k, v in props.items() if v is not None
    }
    return props, source


_MATERIALS: dict[str, tuple[dict[str, Any], dict[str, str], str]] = {
    "AISI304": (
        *_fixed(
            {"E": [183e9, 200e9], "nu": 0.27, "rho": 7900.0, "fu": [210e6, 1050e6], "c": 500.0},
            _W + "SAE_304_stainless_steel",
        ),
        "fu from the annealed to the cold-worked state; fy, G, k, alpha not documented in the source.",
    ),
    "Al6061-T6": (
        *_fixed(
            {
                "E": 68e9,
                "nu": 0.33,
                "rho": 2700.0,
                "fy": 240e6,
                "fu": 290e6,
                "k": [151.0, 202.0],
                "c": 897.0,
                "alpha": 2.32e-5,
            },
            _W + "6061_aluminium_alloy",
        ),
        "minimum fy and fu of the T6 temper (typical 270 and 310 MPa); E = 69 GPa in the text of the source.",
    ),
    "Ti6Al4V": (
        *_fixed(
            {
                "E": [104e9, 113e9],
                "G": [40e9, 45e9],
                "nu": [0.31, 0.37],
                "rho": [4429.0, 4512.0],
                "fy": [880e6, 920e6],
                "fu": [900e6, 950e6],
                "k": [6.7, 7.5],
            },
            _W + "Ti-6Al-4V",
        ),
        "c and alpha not documented in the source.",
    ),
    "Cu": (
        *_fixed(
            {
                "E": [110e9, 128e9],
                "G": 48e9,
                "nu": 0.34,
                "rho": 8935.0,
                "k": 401.0,
                "c": 384.603,
                "alpha": 16.64e-6,
            },
            _W + "Copper",
        ),
        "Pure copper; fy and fu depend on the temper and are not in the source.",
    ),
    "C25/30": (
        *_fixed(
            {"fck": 25e6, "fcm": 33e6, "fctm": 2.56e6, "Ecm": 31476e6, "unit_weight": [24e3, 25e3]},
            _EC2,
            ("fck", "fcm", "fctm", "Ecm", "unit_weight"),
        ),
        "EN 1992-1-1 table 3.1; weight per unit volume 24 kN/m3 without reinforcement, 25 kN/m3 reinforced.",
    ),
    "C24": (
        *_fixed(
            {"fm_k": 24e6, "E0_mean": 11e9, "rho_mean": 420.0},
            _EN338,
            ("fm_k", "E0_mean", "rho_mean"),
        ),
        "Softwood EN 338, class C24.",
    ),
    "PLA": (
        *_fixed({"rho": [1210.0, 1430.0]}, _W + "Polylactic_acid"),
        "Only the density is documented in the source.",
    ),
    "ABS": (
        *_fixed(
            {"rho": [900.0, 1530.0], "k": 0.1, "alpha": 1.2e-4},
            _W + "Acrylonitrile_butadiene_styrene",
        ),
        "Strengths and modulus not documented in the source.",
    ),
}
_NAMES = (*_BANDS, "C45+QT", *_MATERIALS)


@op("material_lookup", ("material", "t"))
def _material(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    name = p.get("material")
    if name not in _NAMES:
        raise ToolError("INVALID_ARGUMENT", f"params.material: one of {', '.join(_NAMES)}.")
    t = qty(p, "t", "length", need=False)
    if name in _MATERIALS:
        if t is not None:
            raise ToolError("INVALID_ARGUMENT", "params.t holds only for steels.")
        props, source, note = _MATERIALS[name]
        props, source = dict(props), dict(source)
    elif name == "C45+QT":
        if t is not None and not 0.016 < t <= 0.040:
            raise ToolError("INVALID_ARGUMENT", "C45+QT documented only for 16 < d <= 40 mm.")
        if t is None:
            ctx.warn("Without t: values of the band 16 < d <= 40 mm, the only documented one.")
        props, source = _fixed(
            _C45_COMMON | {"fy": 430e6, "fu": [650e6, 800e6]},
            {**dict.fromkeys(_C45_COMMON, _OVAKO), "fy": _SWISS, "fu": _SWISS},
        )
        note = "Quenched and tempered (+QT), EN 10277; strengths for 16 < d <= 40 mm."
    else:
        src, bands = _BANDS[str(name)]
        if t is None:
            ctx.warn(
                f"Without t: values for the smallest thickness (up to {fmt_float(bands[0][0] * 1000, 3)} mm)."
            )
            band = bands[0]
        else:
            band = next((b for b in bands if t <= b[0]), None)
            if band is None:
                raise ToolError(
                    "INVALID_ARGUMENT",
                    f"{name}: values documented up to {fmt_float(bands[-1][0] * 1000, 3)} mm of thickness.",
                )
        common, csrc = (_C45_COMMON, _OVAKO) if name == "C45+N" else (_STEEL_COMMON, _EC3)
        props, source = _fixed(
            common | {"fy": band[1] * MPA, "fu": band[2] * MPA},
            {**dict.fromkeys(common, csrc), "fy": src, "fu": src},
        )
        note = (
            "Normalised (+N); minimum fy and fu."
            if name == "C45+N"
            else "Steel EN 10025-2; nominal fy and fu of EN 1993-1-1 table 3.1; k and c not included."
        )
    ctx.formula = "table values with a source for each property (no computation)"
    ctx.units = {f"properties.{k}": _UNITS_PROP[k] for k, v in props.items() if v is not None}
    shown = ", ".join(f"{k} = {v}" for k, v in props.items() if v is not None)
    ctx.formatted = f"{name}: {shown}."
    return {"material": name, "properties": props, "source": source, "note": note}
