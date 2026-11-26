"""D105: math_solve.born_scattering (first Born approximation for a central potential V(r)).

f(q) = -(2 mu / hbar^2) (1 / q) integral_0^oo r sin(q r) V(r) dr, dsigma/dOmega = |f|^2, q = 2 k sin(theta / 2).

References: the textbook amplitudes of the Yukawa potential 2 mu V0 a^2 / (hbar^2 (1 + a^2 q^2)), of the
gaussian sqrt(pi) mu V0 a^3 exp(-a^2 q^2 / 4) / (2 hbar^2) and of the square well
2 mu V0 (sin qR - qR cos qR) / (hbar^2 q^3); the Coulomb potential gives the Rutherford cross section of
eng_calc.nuclear_scattering with E = hbar^2 k^2 / (2 mu); other potentials, and the numbers of 'at', against
the Fourier integral of QUADPACK (scipy quad with weight 'sin', a method other than the mpmath.quadosc of the
server).
"""

from __future__ import annotations

import math
import time
from fractions import Fraction
from typing import Any

import mpmath
import pytest
import sympy as sp
from scipy.integrate import quad

from oce_math_mcp.envelope import Ctx
from oce_math_mcp.errors import ToolError
from oce_math_mcp.expr import parse
from oce_math_mcp.numfunc import compile_mp
from oce_math_mcp.tools import eng_calc, math_solve
from oce_math_mcp.tools.common import grammar

# QUADPACK warns on the cycles where an exponential tail is already below the double range: harmless here,
# every value is compared with the server at 1e-10.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Bad integrand behavior:scipy.integrate.IntegrationWarning"
)

VALS = {
    "mu": 1.3,
    "hbar": 0.7,
    "V0": 2.1,
    "a": 0.9,
    "R": 1.7,
    "alpha": 0.6,
    "q": 1.9,
    "k": 1.4,
    "theta": 1.1,
}


def run(ctx: Ctx | None = None, **args: Any) -> dict:
    return math_solve.handle({"operation": "born_scattering", **args}, ctx or Ctx())


def num(text: str, **values: float) -> float:
    e = parse(text, grammar())
    subs = {s: sp.Float(repr(values[s.name]), 30) for s in e.free_symbols}
    z = complex(sp.N(e.subs(subs), 30))
    assert abs(z.imag) <= 1e-25 * (1 + abs(z.real))
    return z.real


def approx(v: float, rel: float = 1e-12) -> Any:
    return pytest.approx(v, rel=rel, abs=1e-300)


def born_numeric(v: Any, q: float, mu: float, hbar: float, points: tuple[float, ...] = ()) -> float:
    """-(2 mu / hbar^2) (1/q) integral of r sin(q r) V(r): QUADPACK QAGP on [0, 4] (breaks at `points`) plus
    QAWF on [4, oo)."""
    head = quad(
        lambda r: r * math.sin(q * r) * v(r),
        0,
        4,
        points=points or None,
        epsabs=0,
        epsrel=1e-13,
        limit=200,
    )[0]
    tail = quad(lambda r: r * v(r), 4, math.inf, weight="sin", wvar=q, epsabs=1e-15, limlst=200)[0]
    return -2 * mu / hbar**2 / q * (head + tail)


@pytest.mark.parametrize("q", [0.4, 1.9, 6.0])
def test_yukawa(q: float) -> None:
    out = run(potential_energy="-V0*exp(-r/a)/r")
    w = {**VALS, "q": q}
    mu, hb, v0, a = w["mu"], w["hbar"], w["V0"], w["a"]
    want = 2 * mu * v0 * a * a / (hb * hb * (1 + a * a * q * q))
    assert num(out["amplitude"], **w) == approx(want)
    assert num(out["dsigma_dOmega"], **w) == approx(want * want)
    assert out["verified"] is True
    assert out["regularized"] is False


def test_gaussian_and_square_well() -> None:
    w = VALS
    mu, hb, v0, a, rr, q = w["mu"], w["hbar"], w["V0"], w["a"], w["R"], w["q"]
    g = run(potential_energy="-V0*exp(-r**2/a**2)")
    want_g = math.sqrt(math.pi) * mu * v0 * a**3 * math.exp(-(a * a * q * q) / 4) / (2 * hb * hb)
    assert num(g["amplitude"], **w) == approx(want_g)
    s = run(potential_energy="-V0*Heaviside(R - r)")
    want_s = 2 * mu * v0 * (math.sin(q * rr) - q * rr * math.cos(q * rr)) / (hb * hb * q**3)
    assert num(s["amplitude"], **w) == approx(want_s)
    assert g["verified"] is True and s["verified"] is True


def test_theta_form_uses_q_equal_2k_sin_half_theta() -> None:
    out = run(potential_energy="-V0*exp(-r/a)/r")
    w = VALS
    q = 2 * w["k"] * math.sin(w["theta"] / 2)
    assert num(out["amplitude_theta"], **w) == approx(num(out["amplitude"], **{**w, "q": q}))
    assert num(out["dsigma_dOmega_theta"], **w) == approx(
        num(out["dsigma_dOmega"], **{**w, "q": q})
    )


def test_coulomb_is_regularized_and_gives_rutherford() -> None:
    ctx = Ctx()
    out = run(ctx, potential_energy="alpha/r")
    w = VALS
    assert out["regularized"] is True
    assert any("exp(-epsilon*r)" in m for m in ctx.warnings)
    assert num(out["amplitude"], **w) == approx(
        -2 * w["mu"] * w["alpha"] / (w["hbar"] ** 2 * w["q"] ** 2)
    )
    # the same cross section as eng_calc.nuclear_scattering rutherford, SI numbers
    si = {
        "alpha": 2 * 79 * 2.307077e-28,
        "mu": 6.4e-27,
        "hbar": 1.0545718176461565e-34,
        "k": 1.2e15,
    }
    vals = run(potential_energy="alpha/r", at={**si, "theta": math.radians(60)})["values"]
    e = (si["hbar"] * si["k"]) ** 2 / (2 * si["mu"])
    ref = eng_calc.handle(
        {
            "operation": "nuclear_scattering",
            "params": {"mode": "rutherford", "E": e, "alpha": si["alpha"], "theta_deg": 60},
        },
        Ctx(),
    )
    assert vals["dsigma_dOmega"] == approx(ref["dsigma_dOmega"], 1e-12)
    assert vals["q"] == approx(2 * si["k"] * math.sin(math.radians(30)))


@pytest.mark.parametrize("q", [0.6, 2.5])
def test_closed_form_with_piecewise(q: float) -> None:
    """-V0 a^2 sin(r/a) / r^2: SymPy gives atanh(a q) or atanh(1 / (a q)) on either side of a q = 1."""
    v_text = "-V0*a**2*sin(r/a)/r**2"
    out = run(potential_energy=v_text)
    assert out["verified"] is True
    assert "Piecewise" in out["amplitude"]
    want = born_numeric(lambda r: -2.1 * 0.81 * math.sin(r / 0.9) / r**2, q, 1.3, 0.7)
    # the Piecewise is outside the grammar of the server (and sympify is blocked): the SymPy object instead
    sy = math_solve._born_symbols()
    v0, a = sp.Symbol("V0", positive=True), sp.Symbol("a", positive=True)
    rr, qq = sy["r"], sy["q"]
    integral = math_solve._born_integral(-v0 * a**2 * sp.sin(rr / a) / rr**2, rr, qq)
    got = -2 * 1.3 / (0.7**2 * q) * float(sp.N(integral.subs({v0: 2.1, a: 0.9, qq: q}), 30))
    assert got == approx(want, 1e-10)
    at = {"V0": 2.1, "a": 0.9, "mu": 1.3, "hbar": 0.7, "q": q}
    assert run(potential_energy=v_text, at=at)["values"]["f"] == approx(want, 1e-10)


@pytest.mark.parametrize(
    ("v_text", "v_fun"),
    [
        ("-V0*r**2*exp(-r/a)/a**2", lambda r: -2.1 * r * r * math.exp(-r / 0.9) / 0.81),
        ("V0*exp(-r/a)*cos(r/a)", lambda r: 2.1 * math.exp(-r / 0.9) * math.cos(r / 0.9)),
        ("-V0*exp(-r/a)/r", lambda r: -2.1 * math.exp(-r / 0.9) / r),
        ("-V0*Heaviside(R - r)", lambda r: -2.1 if r < 1.7 else 0.0),
    ],
)
def test_values_against_an_independent_integral(v_text: str, v_fun: Any) -> None:
    at = {"V0": 2.1, "a": 0.9, "mu": 1.3, "hbar": 0.7, "q": 1.9}
    if "R" in v_text:
        at = {"V0": 2.1, "R": 1.7, "mu": 1.3, "hbar": 0.7, "q": 1.9}
    out = run(potential_energy=v_text, at=at)
    want = born_numeric(v_fun, 1.9, 1.3, 0.7, (1.7,) if "R" in v_text else ())
    assert out["values"]["f"] == approx(want, 1e-10)
    assert out["values"]["dsigma_dOmega"] == approx(want * want, 1e-10)
    assert "amplitude" not in out


def test_woods_saxon_has_no_closed_form_but_a_number_with_at() -> None:
    """The nuclear potential -V0 / (1 + exp((r - R) / a)): SymPy finds no closed form, 'at' gives the number."""
    v_text = "-V0/(1 + exp((r - R)/a))"
    t0 = time.perf_counter()
    with pytest.raises(ToolError, match="'at'") as err:
        run(potential_energy=v_text)
    # SymPy's default strategy took 16.03 s to give up here, over the 5 s limit of a call; Meijer G 0.10 s
    assert time.perf_counter() - t0 < 5
    assert err.value.code == "MATH_ERROR"
    at = {"V0": 2.1, "R": 3.0, "a": 0.5, "mu": 1.3, "hbar": 0.7, "q": 1.9}
    out = run(potential_energy=v_text, at=at)
    want = born_numeric(lambda r: -2.1 / (1 + math.exp((r - 3.0) / 0.5)), 1.9, 1.3, 0.7)
    assert out["values"]["f"] == approx(want, 1e-10)


@pytest.mark.parametrize(
    ("args", "code", "match"),
    [
        ({}, "INVALID_ARGUMENT", "potential_energy"),
        ({"potential_energy": "V0"}, "INVALID_ARGUMENT", "depend on r"),
        ({"potential_energy": "-V0*exp(-q*r)/r"}, "INVALID_ARGUMENT", "'q'"),
        ({"potential_energy": "-V0*exp(-mu*r)/r"}, "INVALID_ARGUMENT", "'mu'"),
        ({"potential_energy": "V0*a**2/r**3"}, "MATH_ERROR", "does not converge"),
        (
            {
                "potential_energy": "V0*a**2/r**3",
                "at": {"V0": 1, "a": 1, "mu": 1, "hbar": 1, "q": 1},
            },
            "MATH_ERROR",
            "r -> 0",
        ),
        ({"potential_energy": "V0*r**2/a**2"}, "MATH_ERROR", "at least as 1/r"),
        (
            {
                "potential_energy": "-V0*exp(-r/a)/r",
                "at": {"V0": 1, "a": 1, "mu": 1, "hbar": 1, "q": 1, "theta": 1},
            },
            "INVALID_ARGUMENT",
            "q",
        ),
        (
            {
                "potential_energy": "-V0*exp(-r/a)/r",
                "at": {"V0": 1, "a": 0, "mu": 1, "hbar": 1, "q": 1},
            },
            "INVALID_ARGUMENT",
            "> 0",
        ),
        (
            {
                "potential_energy": "-V0*exp(-r/a)/r",
                "at": {"V0": 1, "a": 1, "mu": 1, "hbar": 1, "k": 1, "theta": 4},
            },
            "INVALID_ARGUMENT",
            "theta",
        ),
        (
            {"potential_energy": "-V0*exp(-r/a)/r", "at": {"V0": 1, "a": 1, "mu": 1, "hbar": 1}},
            "INVALID_ARGUMENT",
            "q",
        ),
        (
            {
                "potential_energy": "-V0*exp(-r/a)/r",
                "at": {"V0": 1, "a": 1, "mu": 1, "hbar": 1, "q": 1, "k": 1, "theta": 1},
            },
            "INVALID_ARGUMENT",
            "q",
        ),
        (
            {"potential_energy": "-V0*exp(-r/a)/r", "at": {"V0": 1, "mu": 1, "hbar": 1, "q": 1}},
            "INVALID_ARGUMENT",
            "'a'",
        ),
    ],
)
def test_errors(args: dict, code: str, match: str) -> None:
    with pytest.raises(ToolError, match=match) as err:
        run(**args)
    assert err.value.code == code


def test_woods_saxon_numbers_within_the_time_limit() -> None:
    """The example of the error message: 17.68 s before compile_mp had guard digits (limit 5 s per call)."""
    at = {"V0": 50, "R": 4, "a": 0.6, "mu": 1, "hbar": 1, "q": 1}
    t0 = time.perf_counter()
    out = run(potential_energy="-V0/(1 + exp((r - R)/a))", at=at)
    assert time.perf_counter() - t0 < 5
    want = born_numeric(lambda r: -50 / (1 + math.exp((r - 4) / 0.6)), 1.0, 1.0, 1.0)
    assert out["values"]["f"] == approx(want, 1e-10)


def test_compile_mp_has_no_jump_where_the_exact_sum_starts() -> None:
    """5x/3 - 20/3 near x = 4 (the exponent of Woods-Saxon): the rounded path and the exact path of D104
    differed by up to 50 units in the last place where the exact one starts, so mpmath.quadosc could not
    converge. With guard digits every value is the exact one rounded (2 units in the last place at most)."""
    x = sp.Symbol("x")
    e = sp.Rational(5, 3) * x - sp.Rational(20, 3)
    with mpmath.workdps(20):
        f = compile_mp(e, [x], 20)
        worst = mpmath.mpf(0)
        for k in range(1, 400):
            xv = (
                mpmath.mpf(4) + mpmath.mpf(k) / 2000
            )  # |x - 4| from 5e-4 to 0.2: the switch is near 0.08
            got = f(xv)
            assert int(got.man).bit_length() <= mpmath.mp.prec  # rounded to the caller's precision
            man, exp = int(xv.man), int(xv.exp)
            xf = Fraction(man) * 2**exp if exp >= 0 else Fraction(man, 2**-exp)
            exact = Fraction(5, 3) * xf - Fraction(20, 3)
            with mpmath.workdps(60):
                rel = abs(got - mpmath.mpf(exact.numerator) / exact.denominator) / abs(got)
            worst = max(worst, rel)
        assert worst <= mpmath.mpf(2) ** -69, worst  # 20 digits = 70 bits
