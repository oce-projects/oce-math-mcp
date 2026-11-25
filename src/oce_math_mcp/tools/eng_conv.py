"""eng_calc: convection coefficient from the Nusselt correlations (D52).

Correlations (sources: Wikipedia «Nusselt number», «Churchill–Bernstein equation»,
«Heat transfer coefficient», «Reynolds number», which cite Incropera and the original papers):
- forced convection: laminar flat plate (Nu_L = 0.664 Re^1/2 Pr^1/3, transition at Re_x about
  5e5), cylinder in crossflow (Churchill-Bernstein), sphere (Faeth), pipe (fully developed laminar
  3.66 or 4.36; turbulent Gnielinski with the Filonenko f, Dittus-Boelter, Sieder-Tate);
- natural convection: vertical wall (Churchill-Chu), horizontal plate (Incropera p. 577),
  horizontal cylinder (Churchill-Chu), sphere (Yuge), horizontal layer heated from below
  (Globe-Dropkin).
The fluid properties are given by the caller (k, Pr, nu or mu and rho, beta), at the film temperature.
There is no plate in turbulent flow: no verified source found (D52).
"""

from __future__ import annotations

import math
from typing import Any

from ..envelope import Ctx
from ..errors import ToolError
from .eng_base import choice, fmt_sig, need, only, op, qty, temperature, units_for

G0 = 9.80665

_COMMON = {"k", "Pr", "nu", "mu", "rho", "A", "T_s", "T_inf", "dT"}
_FORCED = _COMMON | {"v", "Re"}
_NATURAL = _COMMON | {"Ra", "beta", "g"}
_GEO: dict[str, set[str]] = {
    "flat_plate": _FORCED | {"L"},
    "cylinder_crossflow": _FORCED | {"D"},
    "sphere_forced": _FORCED | {"D"},
    "pipe": _FORCED | {"D", "L", "bc", "correlation", "mu_w", "heating"},
    "vertical_plate": _NATURAL | {"L"},
    "horizontal_plate": _NATURAL | {"L", "P", "facing"},
    "horizontal_cylinder": _NATURAL | {"D"},
    "sphere_natural": _NATURAL | {"D"},
    "horizontal_enclosure": _NATURAL | {"L"},
}
_UNITS = {
    "Re": "1",
    "Gr": "1",
    "Ra": "1",
    "Pr": "1",
    "Nu": "1",
    "h": "W/(m2*K)",
    "L_char": "m",
    "f_darcy": "1",
    "viscosity_ratio": "1",
    "h_local_end": "W/(m2*K)",
    "Q": "W",
    "q_flux": "W/m2",
    "beta": "1/K",
}


def _nu(p: dict[str, Any]) -> float:
    if p.get("nu") is not None:
        if p.get("mu") is not None or p.get("rho") is not None:
            raise ToolError("INVALID_ARGUMENT", "Give nu or mu and rho, not all three.")
        return need(p, "nu", "kinematic_viscosity")
    if p.get("mu") is None or p.get("rho") is None:
        raise ToolError("INVALID_ARGUMENT", "The viscosity is needed: nu, or mu and rho.")
    return need(p, "mu", "dynamic_viscosity") / need(p, "rho", "density")


def _dT(p: dict[str, Any], ctx: Ctx) -> tuple[float | None, float | None]:
    """(T_s - T_inf, film temperature) if present; dT alone does not give the film."""
    ts, ti = temperature(p, "T_s", need=False), temperature(p, "T_inf", need=False)
    if (ts is None) != (ti is None):
        raise ToolError("INVALID_ARGUMENT", "T_s and T_inf must be given together.")
    if ts is not None and ti is not None:
        if p.get("dT") is not None:
            raise ToolError("INVALID_ARGUMENT", "Give dT or T_s and T_inf, not all three.")
        return ts - ti, (ts + ti) / 2
    if p.get("dT") is not None:
        return need(p, "dT", None, positive=False), None
    return None, None


def _length(p: dict[str, Any], key: str) -> float:
    return need(p, key, "length")


def _range(ctx: Ctx, name: str, x: float, lo: float, hi: float, what: str) -> None:
    if not lo <= x <= hi:
        ctx.warn(
            f"{what} = {fmt_sig(x, 4)} outside the range of {name} ({fmt_sig(lo, 3)} - {fmt_sig(hi, 3)})."
        )


@op("convection_correlation", ("geometry", *sorted(set().union(*_GEO.values()))))
def _convection(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    geo = choice(p, "geometry", tuple(_GEO))
    only(p, geo, _GEO[geo], field="geometry")
    k = need(p, "k", "thermal_conductivity")
    pr = need(p, "Pr", None)
    dT, t_film = _dT(p, ctx)
    out: dict[str, Any] = {"Pr": pr}
    if geo in ("flat_plate", "cylinder_crossflow", "sphere_forced", "pipe"):
        L = _length(p, "L" if geo == "flat_plate" else "D")
        if (p.get("Re") is None) == (p.get("v") is None):
            raise ToolError("INVALID_ARGUMENT", "Forced convection needs one of v and Re.")
        re = (
            need(p, "Re", None)
            if p.get("Re") is not None
            else need(p, "v", "velocity") * L / _nu(p)
        )
        out["Re"] = re
        nu_num = _forced(geo, re, pr, p, L, dT, out, ctx)
    else:
        L = _natural_length(geo, p)
        nu_num = _natural(geo, p, pr, L, dT, t_film, out, ctx)
    h = nu_num * k / L
    out |= {"Nu": nu_num, "h": h, "L_char": L}
    if dT is not None:
        out["q_flux"] = h * dT
        area = qty(p, "A", "area", need=False)
        if area is not None:
            out["Q"] = h * area * dT
    elif p.get("A") is not None:
        ctx.warn("With A also dT (or T_s and T_inf) is needed for the heat Q.")
    ctx.units = units_for(out, _UNITS)
    ctx.formatted = (
        f"{out['correlation']}: Nu = {fmt_sig(nu_num, 5)}, h = {fmt_sig(h, 5)} W/(m2 K)"
        + (f", Q = {fmt_sig(out['Q'], 5)} W" if "Q" in out else "")
        + "."
    )
    return out


def _forced(
    geo: str,
    re: float,
    pr: float,
    p: dict[str, Any],
    L: float,
    dT: float | None,
    out: dict[str, Any],
    ctx: Ctx,
) -> float:
    if geo == "flat_plate":
        if pr <= 0.6:
            ctx.warn("Laminar plate: the correlation holds for Pr > 0.6.")
        if re > 5e5:
            ctx.warn(
                "Re_L above about 5e5: the boundary layer becomes turbulent, the laminar correlation does not hold."
            )
        nu = 0.664 * re**0.5 * pr ** (1 / 3)
        out |= {
            "correlation": "laminar flat plate, Nu_L = 0.664 Re^1/2 Pr^1/3",
            "regime": "laminar",
        }
        out["h_local_end"] = (
            0.332 * re**0.5 * pr ** (1 / 3) * need(p, "k", "thermal_conductivity") / L
        )
        ctx.formula = "Nu_L = 0,664 Re_L^(1/2) Pr^(1/3) (media); Nu_x = 0,332 Re_x^(1/2) Pr^(1/3) (locale); h = Nu k / L"
        return nu
    if geo == "cylinder_crossflow":
        if re * pr < 0.2:
            ctx.warn("Churchill-Bernstein holds for Re Pr >= 0.2.")
        nu = 0.3 + 0.62 * re**0.5 * pr ** (1 / 3) / (1 + (0.4 / pr) ** (2 / 3)) ** 0.25 * (
            1 + (re / 282000) ** (5 / 8)
        ) ** (4 / 5)
        out |= {
            "correlation": "cylinder in crossflow, Churchill-Bernstein",
            "regime": "forced",
        }
        ctx.formula = "Nu_D = 0.3 + 0.62 Re^(1/2) Pr^(1/3) / [1 + (0.4/Pr)^(2/3)]^(1/4) [1 + (Re/282000)^(5/8)]^(4/5)"
        return nu
    if geo == "sphere_forced":
        _range(ctx, "Faeth", re, 0.0, 1800.0, "Re_D")
        nu = 2 + 0.555 * re**0.5 * pr ** (1 / 3) / (1 + 1.232 / (re * pr ** (4 / 3))) ** 0.5
        out |= {"correlation": "sphere in forced flow, Faeth", "regime": "forced"}
        ctx.formula = "Nu_D = 2 + 0,555 Re^(1/2) Pr^(1/3) / [1 + 1,232 / (Re Pr^(4/3))]^(1/2)"
        return nu
    return _pipe(re, pr, p, L, dT, out, ctx)


def _pipe(
    re: float,
    pr: float,
    p: dict[str, Any],
    D: float,
    dT: float | None,
    out: dict[str, Any],
    ctx: Ctx,
) -> float:
    corr = choice(
        p, "correlation", ("gnielinski", "dittus_boelter", "sieder_tate"), default="gnielinski"
    )
    if p.get("L") is not None and _length(p, "L") / D < 10:
        ctx.warn(
            "L/D below 10: the flow is not developed, the correlations underestimate the heat transfer at the inlet."
        )
    if re < 2300:
        bc = choice(p, "bc", ("uniform_T", "uniform_q"), default="uniform_T")
        nu = 3.66 if bc == "uniform_T" else 4.36
        wall = "T" if bc == "uniform_T" else "flux"
        out |= {
            "correlation": f"fully developed laminar pipe, Nu = {fmt_sig(nu, 3)} (uniform wall {wall})",
            "regime": "laminar",
        }
        ctx.formula = "Fully developed laminar: Nu_D = 3.66 (uniform wall T), 4.36 (uniform flux)"
        return nu
    if p.get("bc") is not None:
        ctx.warn("bc is used only for laminar flow (Re < 2300): ignored.")
    out["regime"] = "turbulent" if re >= 10000 else "transitional"
    if corr == "gnielinski":
        _range(ctx, "Gnielinski", re, 3000.0, 5e6, "Re_D")
        _range(ctx, "Gnielinski", pr, 0.5, 2000.0, "Pr")
        f = (0.79 * math.log(re) - 1.64) ** -2
        nu = (f / 8) * (re - 1000) * pr / (1 + 12.7 * (f / 8) ** 0.5 * (pr ** (2 / 3) - 1))
        out |= {"correlation": "turbulent pipe, Gnielinski with the Filonenko f", "f_darcy": f}
        ctx.formula = "f = (0.79 ln Re - 1.64)^-2; Nu = (f/8)(Re - 1000) Pr / [1 + 12.7 (f/8)^(1/2) (Pr^(2/3) - 1)]"
        return nu
    if corr == "dittus_boelter":
        _range(ctx, "Dittus-Boelter", re, 10000.0, math.inf, "Re_D")
        _range(ctx, "Dittus-Boelter", pr, 0.6, 160.0, "Pr")
        heating = p.get("heating")
        if heating is None:
            if dT is None:
                raise ToolError(
                    "INVALID_ARGUMENT",
                    "Dittus-Boelter: give heating (true if the fluid is heated) or T_s and T_inf.",
                )
            heating = dT > 0
        if not isinstance(heating, bool):
            raise ToolError("INVALID_ARGUMENT", "params.heating is true or false.")
        n = 0.4 if heating else 0.3
        nu = 0.023 * re**0.8 * pr**n
        out["correlation"] = f"turbulent pipe, Dittus-Boelter (n = {n})"
        ctx.formula = (
            "Nu = 0.023 Re^(4/5) Pr^n; n = 0.4 if the fluid is heated, 0.3 if it is cooled"
        )
        return nu
    _range(ctx, "Sieder-Tate", re, 10000.0, math.inf, "Re_D")
    _range(ctx, "Sieder-Tate", pr, 0.7, 16700.0, "Pr")
    if p.get("mu") is None or p.get("mu_w") is None:
        raise ToolError(
            "INVALID_ARGUMENT", "Sieder-Tate needs mu (at the bulk) and mu_w (at the wall)."
        )
    ratio = need(p, "mu", "dynamic_viscosity") / need(p, "mu_w", "dynamic_viscosity")
    nu = 0.027 * re**0.8 * pr ** (1 / 3) * ratio**0.14
    out |= {"correlation": "turbulent pipe, Sieder-Tate", "viscosity_ratio": ratio}
    ctx.formula = "Nu = 0.027 Re^(4/5) Pr^(1/3) (mu_b / mu_w)^0.14"
    return nu


def _natural_length(geo: str, p: dict[str, Any]) -> float:
    if geo in ("horizontal_cylinder", "sphere_natural"):
        return _length(p, "D")
    if geo == "horizontal_plate":
        if p.get("L") is not None:
            if p.get("A") is not None and p.get("P") is not None:
                raise ToolError(
                    "INVALID_ARGUMENT", "Horizontal plate: give L or A and P (L = A / P)."
                )
            return _length(p, "L")
        if p.get("A") is None or p.get("P") is None:
            raise ToolError(
                "INVALID_ARGUMENT",
                "Horizontal plate: L = A / P is needed, or A (area) and P (perimeter).",
            )
        return need(p, "A", "area") / _length(p, "P")
    return _length(p, "L")


def _natural(
    geo: str,
    p: dict[str, Any],
    pr: float,
    L: float,
    dT: float | None,
    t_film: float | None,
    out: dict[str, Any],
    ctx: Ctx,
) -> float:
    if p.get("Ra") is not None:
        ra = need(p, "Ra", None)
    else:
        if dT is None:
            raise ToolError(
                "INVALID_ARGUMENT",
                "Natural convection: Ra is needed, or dT (or T_s and T_inf) with the properties.",
            )
        beta = qty(p, "beta", "thermal_expansion", need=False)
        if beta is None:
            if t_film is None:
                raise ToolError(
                    "INVALID_ARGUMENT",
                    "beta is needed, or T_s and T_inf for beta = 1 / T_film (ideal gas).",
                )
            beta = 1 / t_film
            ctx.warn(
                f"beta = 1 / T_film = {fmt_sig(beta, 4)} 1/K: holds for ideal gases; for liquids give beta."
            )
        g = qty(p, "g", "acceleration", default=G0)
        assert g is not None
        nu_k = _nu(p)
        gr = g * beta * abs(dT) * L**3 / nu_k**2
        ra = gr * pr
        out |= {"Gr": gr, "beta": beta}
    out["Ra"] = ra
    if geo == "vertical_plate":
        psi = 1 + (0.492 / pr) ** (9 / 16)
        if 0.1 < ra < 1e9:
            nu = 0.68 + 0.67 * ra**0.25 / psi ** (4 / 9)
            out |= {"correlation": "vertical wall, laminar Churchill-Chu", "regime": "laminar"}
            ctx.formula = "Nu_L = 0.68 + 0.67 Ra^(1/4) / [1 + (0.492/Pr)^(9/16)]^(4/9)"
        else:
            if ra >= 1e12:
                ctx.warn("Churchill-Chu holds for Ra < 1e12.")
            nu = (0.825 + 0.387 * ra ** (1 / 6) / psi ** (8 / 27)) ** 2
            out |= {
                "correlation": "vertical wall, Churchill-Chu",
                "regime": "turbulent" if ra >= 1e9 else "laminar",
            }
            ctx.formula = "Nu_L = {0,825 + 0,387 Ra^(1/6) / [1 + (0,492/Pr)^(9/16)]^(8/27)}^2"
        return nu
    if geo == "horizontal_plate":
        facing = choice(p, "facing", ("up", "down"))
        if dT is None:
            raise ToolError(
                "INVALID_ARGUMENT",
                "Horizontal plate: the sign of dT (or T_s and T_inf) is needed to know whether it is hot.",
            )
        hot = dT > 0
        if (facing == "up") == hot:  # hot facing up or cold facing down
            if ra <= 1e7:
                _range(ctx, "0.54 Ra^1/4", ra, 1e4, 1e7, "Ra_L")
                nu = 0.54 * ra**0.25
                out["correlation"] = (
                    "horizontal plate, hot surface facing up (or cold facing down), 0.54 Ra^1/4"
                )
            else:
                _range(ctx, "0.15 Ra^1/3", ra, 1e7, 1e11, "Ra_L")
                nu = 0.15 * ra ** (1 / 3)
                out["correlation"] = (
                    "horizontal plate, hot surface facing up (or cold facing down), 0.15 Ra^1/3"
                )
        else:
            _range(ctx, "0.27 Ra^1/4", ra, 1e5, 1e10, "Ra_L")
            nu = 0.27 * ra**0.25
            out["correlation"] = (
                "horizontal plate, hot surface facing down (or cold facing up), 0.27 Ra^1/4"
            )
        ctx.formula = (
            "L = A / P; Nu_L = 0.54 Ra^(1/4), 0.15 Ra^(1/3) or 0.27 Ra^(1/4) (Incropera p. 577)"
        )
        return nu
    if geo == "horizontal_cylinder":
        _range(ctx, "Churchill-Chu", ra, 1e-5, 1e12, "Ra_D")
        nu = (0.6 + 0.387 * ra ** (1 / 6) / (1 + (0.559 / pr) ** (9 / 16)) ** (8 / 27)) ** 2
        out["correlation"] = "horizontal cylinder, Churchill-Chu"
        ctx.formula = "Nu_D = {0.6 + 0.387 Ra^(1/6) / [1 + (0.559/Pr)^(9/16)]^(8/27)}^2"
        return nu
    if geo == "sphere_natural":
        _range(ctx, "Yuge", ra, 1.0, 1e5, "Ra_D")
        if not 0.5 <= pr <= 2:
            ctx.warn("Yuge holds for Pr about 1 (gases).")
        out["correlation"] = "sphere, Yuge"
        ctx.formula = "Nu_D = 2 + 0.43 Ra^(1/4)"
        return 2 + 0.43 * ra**0.25
    _range(ctx, "Globe-Dropkin", ra, 1.5e5, 6.8e8, "Ra_L")
    _range(ctx, "Globe-Dropkin", pr, 0.02, 8750.0, "Pr")
    out["correlation"] = "horizontal layer heated from below, Globe-Dropkin"
    ctx.formula = "Nu_L = 0.069 Ra^(1/3) Pr^0.074 (L = distance between the plates)"
    return 0.069 * ra ** (1 / 3) * pr**0.074
