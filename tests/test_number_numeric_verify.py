"""number_theory, math_numeric, math_verify."""

from __future__ import annotations

import math
from fractions import Fraction

from tests.conftest import ok


def nt(**kw: object) -> dict:
    return ok("number_theory", **kw)


def test_primality_big_as_string() -> None:
    assert nt(operation="is_prime", n=str(2**127 - 1))["is_prime"] is True
    assert nt(operation="is_prime", n=2**61 - 1)["is_prime"] is True
    assert nt(operation="is_prime", n=2**61 + 1)["is_prime"] is False


def test_factorint_and_divisors() -> None:
    r = nt(operation="factorint", n=360)
    assert r["factors"] == {"2": 3, "3": 2, "5": 1}
    assert math.prod(int(p) ** e for p, e in r["factors"].items()) == 360
    d = nt(operation="divisors", n=28)
    assert d["divisors"] == [1, 2, 4, 7, 14, 28] and d["sum"] == 56


def test_totient_matches_brute_force() -> None:
    assert nt(operation="totient", n=36)["value"] == sum(
        1 for k in range(1, 37) if math.gcd(k, 36) == 1
    )


def test_gcd_ext_bezout_and_steps() -> None:
    r = nt(operation="gcd_ext", n=240, m=46, show_steps=True)
    assert r["gcd"] == 2 and 240 * r["x"] + 46 * r["y"] == 2
    assert "euclid" in " ".join(s["text"] for s in r["steps"]).lower()


def test_modular() -> None:
    assert nt(operation="mod_pow", n=2, k=10**18, m=10**9 + 7)["value"] == pow(2, 10**18, 10**9 + 7)
    inv = nt(operation="mod_inv", n=17, m=3120)["value"]
    assert (17 * inv) % 3120 == 1
    roots = nt(operation="sqrt_mod", n=2, m=7)["roots"]
    assert sorted(roots) == sorted(r for r in range(7) if r * r % 7 == 2)
    k = nt(operation="discrete_log", n=13, k=3, m=17)["value"]
    assert pow(3, k, 17) == 13
    assert nt(operation="legendre", n=2, m=7)["value"] == (1 if pow(2, 3, 7) == 1 else -1)


def test_crt() -> None:
    r = nt(operation="crt", values=[2, 3, 2], moduli=[3, 5, 7])
    v = r["value"]
    assert v % 3 == 2 and v % 5 == 3 and v % 7 == 2 and r["modulus"] == 105


def test_diophantine_linear() -> None:
    r = nt(operation="diophantine", expr="3*x + 5*y = 1", variables=["x", "y"])
    assert r["solutions"]


def test_continued_fraction() -> None:
    r = nt(operation="continued_fraction", expr="415/93")
    frac = Fraction(r["terms"][-1])
    for a in reversed(r["terms"][:-1]):
        frac = a + 1 / frac
    assert frac == Fraction(415, 93)
    r = nt(operation="continued_fraction", expr="sqrt(7)")
    assert r["periodic"] is True and r["terms"][0] == 2


def test_partitions_matches_dp() -> None:
    p = [1] + [0] * 100
    for part in range(1, 101):
        for k in range(part, 101):
            p[k] += p[k - part]
    assert nt(operation="partitions", n=100)["value"] == p[100]


def test_primes_and_digits() -> None:
    assert nt(operation="primerange", n=10, m=30)["primes"] == [11, 13, 17, 19, 23, 29]
    assert nt(operation="nth_prime", n=1)["value"] == 2
    assert nt(operation="digits", n=2**100)["digit_sum"] == sum(map(int, str(2**100)))
    assert nt(operation="gcd", values=[84, 36, 120])["value"] == math.gcd(84, 36, 120)


# --- math_numeric ---------------------------------------------------------


def num(**kw: object) -> dict:
    return ok("math_numeric", **kw)


def test_evalf_pi_50_digits() -> None:
    assert num(operation="evalf", expr="pi", digits=50)["value"].startswith(
        "3.141592653589793238462643383279502884197169399375"
    )


def test_quad_gaussian() -> None:
    r = num(operation="quad", expr="exp(-x**2)", variables=["x"], bounds=["-oo", "oo"])
    assert abs(float(r["value"]) - math.sqrt(math.pi)) < 1e-12


def test_root_dottie() -> None:
    r = num(operation="root", expr="cos(x) = x", variables=["x"], bounds=[0, 1])
    xv = r["x"]["x"]
    assert abs(math.cos(xv) - xv) < 1e-12


def test_minimize_rosenbrock() -> None:
    r = num(
        operation="minimize",
        expr="(1 - x)**2 + 100*(y - x**2)**2",
        variables=["x", "y"],
        x0=[-1, 1],
    )
    assert abs(r["x"]["x"] - 1) < 1e-4 and abs(r["x"]["y"] - 1) < 1e-4


def test_linprog_and_milp() -> None:
    r = num(
        operation="linprog",
        expr="3*x + 2*y",
        constraints=["x + y <= 4", "x + 3*y <= 6", "x >= 0", "y >= 0"],
        variables=["x", "y"],
        direction="maximize",
    )
    assert abs(r["fun"] - 12) < 1e-9
    r = num(
        operation="linprog",
        expr="x + y",
        constraints=["2*x + 2*y >= 3", "x >= 0", "y >= 0"],
        variables=["x", "y"],
        integer=True,
    )
    assert abs(r["fun"] - 2) < 1e-9


def test_ode_ivp_exponential_decay() -> None:
    r = num(operation="ode_ivp", expr="-y", variables=["t", "y"], x0=[1], bounds=[0, 1])
    assert abs(r["final"][0] - math.exp(-1)) < 1e-8


def test_polyroots_high_precision() -> None:
    roots = num(operation="polyroots", expr="x**2 - 2", variables=["x"], digits=30)["roots"]
    assert any(r.startswith("1.41421356237309504880168872") for r in roots)


def test_identify_golden_ratio() -> None:
    r = num(operation="identify", expr="1.6180339887498948482")
    assert any("sqrt(5)" in c for c in r["candidates"] + [r["nsimplify"]])


def test_curve_fit_exponential() -> None:
    xs = [0, 0.5, 1, 1.5, 2, 2.5, 3]
    ys = [2 * math.exp(0.5 * v) for v in xs]
    r = num(
        operation="curve_fit", expr="a*exp(b*x)", variables=["x"], data_x=xs, data_y=ys, x0=[1, 1]
    )
    assert abs(r["params"]["a"] - 2) < 1e-6 and abs(r["params"]["b"] - 0.5) < 1e-6


def test_nsum_basel() -> None:
    r = num(operation="nsum", expr="1/n**2", variables=["n"], bounds=[1, "oo"], digits=20)
    assert abs(float(r["value"]) - math.pi**2 / 6) < 1e-15


# --- math_verify ----------------------------------------------------------


def ver(**kw: object) -> dict:
    return ok("math_verify", **kw)


def test_equivalent_symbolic_and_counterexample() -> None:
    r = ver(operation="equivalent", expr="sin(x)**2", expected="1 - cos(x)**2")
    assert r["equivalent"] is True and r["method"] == "symbolic"
    r = ver(operation="equivalent", expr="x + 1", expected="x + 2")
    assert r["equivalent"] is False and r["counterexample"] is not None


def test_check_solution() -> None:
    assert ver(operation="check_solution", expr="x**2 = 4", solution={"x": "2"})["valid"] is True
    assert ver(operation="check_solution", expr="x**2 = 4", solution={"x": "3"})["valid"] is False


def test_check_derivative_and_antiderivative() -> None:
    assert ver(operation="check_derivative", expr="x**3", expected="3*x**2", variables=["x"])[
        "valid"
    ]
    assert ver(
        operation="check_antiderivative", expr="x*log(x) - x", expected="log(x)", variables=["x"]
    )["valid"]


def test_check_numeric() -> None:
    assert ver(operation="check_numeric", expr="sqrt(2)", expected="1.41421356", tolerance=1e-8)[
        "valid"
    ]
    assert not ver(operation="check_numeric", expr="sqrt(2)", expected="1.4142", tolerance=1e-8)[
        "valid"
    ]


def test_check_inequality() -> None:
    r = ver(
        operation="check_inequality",
        expr="(a + b)/2 >= sqrt(a*b)",
        variables=["a", "b"],
        domain="positive",
    )
    assert r["holds_on_samples"] is True
    r = ver(
        operation="check_inequality", expr="a**2 >= a", variables=["a"], domain="real", samples=200
    )
    assert r["holds_on_samples"] is False
    assert 0 < r["counterexample"]["a"] < 1
