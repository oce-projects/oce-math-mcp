"""math_eval: exact arithmetic, functions, statistics, Italian finance, units."""

from __future__ import annotations

import math
import statistics
from decimal import Decimal
from fractions import Fraction

import pytest

from tests.conftest import call, err, ok


def val(expr: str, **kw: object) -> str:
    return ok("math_eval", expr=expr, **kw)["value"]


def test_decimal_is_exact() -> None:
    r = ok("math_eval", expr="0.1 + 0.2")
    assert r["value"] == "0.3"
    assert r["is_exact"] is True
    assert r["mode"] == "decimal"


def test_fraction_mode() -> None:
    assert val("1/3 + 1/6", exact_mode="fraction") == "1/2"


def test_decimal_precision_rounding_half_up() -> None:
    assert val("1/3") == "0.333333"
    assert val("2/3", precision=2) == "0.67"
    assert val("round(2.5)") == "3"
    assert val("round(2.675, 2)") == "2.68"


def test_float_mode() -> None:
    assert ok("math_eval", expr="1/3", exact_mode="float")["value"] == repr(1 / 3)


def test_irrational_in_fraction_mode_warns() -> None:
    resp = call("math_eval", expr="sqrt(2)", exact_mode="fraction")
    assert resp["ok"]
    assert resp["result"]["exact"] == "sqrt(2)"
    assert resp["result"]["is_exact"] is False
    assert resp["warnings"]


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("sind(30)", "0.5"),
        ("cosd(60)", "0.5"),
        ("log(8, 2)", "3"),
        ("log10(1000)", "3"),
        ("ln(E)", "1"),
        ("log2(1024)", "10"),
        ("cbrt(27)", "3"),
        ("root(16, 4)", "2"),
        ("abs(-7) + floor(2.7) + ceil(2.1)", "12"),
    ],
)
def test_exact_functions(expr: str, expected: str) -> None:
    assert val(expr) == expected


def test_atan2_exact_form() -> None:
    assert ok("math_eval", expr="atan2(1, 1)")["exact"] == "pi/4"


def test_variables_numbers_and_strings() -> None:
    assert val("a*b + c", variables={"a": 2, "b": "3.5", "c": 1}) == "8"


def test_combinatorics() -> None:
    assert val("comb(10, 3)") == str(math.comb(10, 3))
    assert val("perm(10, 3)") == str(math.perm(10, 3))
    assert val("factorial(20)") == str(math.factorial(20))
    inv = int(val("mod_inv(3, 11)"))
    assert (3 * inv) % 11 == 1
    assert val("gcd(84, 36)") == str(math.gcd(84, 36))
    assert val("lcm(4, 6)") == str(math.lcm(4, 6))


def test_statistics() -> None:
    data = [2, 4, 4, 4, 5, 5, 7, 9]
    assert val("mean([1, 2, 3, 4])") == "2.5"
    assert val("median([3, 1, 2])") == "2"
    assert val("pstdev([2, 4, 4, 4, 5, 5, 7, 9])") == "2"
    expected = Decimal(str(statistics.stdev(data))).quantize(Decimal("0.000001"))
    assert val("stdev([2, 4, 4, 4, 5, 5, 7, 9])") == str(expected).rstrip("0")


def test_distributions() -> None:
    assert val("norm_cdf(0, 0, 1)") == "0.5"
    expected = statistics.NormalDist().inv_cdf(0.975)
    got = float(val("norm_ppf(0.975, 0, 1)", precision=9))
    assert abs(got - expected) < 1e-8
    assert val("binom_pmf(2, 4, 0.5)", exact_mode="fraction") == str(
        Fraction(math.comb(4, 2), 2**4)
    )


def test_italian_vat_and_withholding() -> None:
    assert val("vat_add(100)") == "122"
    assert val("vat_remove(122)") == "100"
    assert val("vat(100, 10)") == "10"
    assert val("withholding(1000)") == "200"


def test_loan_payment() -> None:
    r, n, pv = 0.05 / 12, 360, 200000
    expected = pv * r / (1 - (1 + r) ** -n)
    got = float(val("pmt(0.05/12, 360, 200000)", precision=2))
    assert abs(got - expected) < 0.01


def test_npv_and_irr() -> None:
    expected = -100 + 60 / 1.1 + 60 / 1.1**2
    assert abs(float(val("npv(0.1, [-100, 60, 60])", precision=9)) - expected) < 1e-9
    rate = float(val("irr([-100, 60, 60])", precision=12))
    assert abs(-100 + 60 / (1 + rate) + 60 / (1 + rate) ** 2) < 1e-8


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("unit_convert(1, 'in', 'mm')", "25.4"),
        ("unit_convert(100, 'C', 'F')", "212"),
        ("unit_convert(1, 'CV', 'W')", "735.49875"),
        ("unit_convert(1, 'kWh', 'J')", "3600000"),
        ("unit_convert(1, 'lb', 'kg')", "0.45359237"),
    ],
)
def test_unit_convert(expr: str, expected: str) -> None:
    assert val(expr, precision=10) == expected


def test_unit_convert_psi() -> None:
    psi_pa = Fraction("4.4482216152605") / Fraction("0.0254") ** 2
    got = Fraction(val("unit_convert(1, 'bar', 'psi')", exact_mode="fraction"))
    assert got == Fraction(100000) / psi_pa


def test_unit_convert_wrong_category() -> None:
    assert err("math_eval", expr="unit_convert(1, 'm', 'kg')") == "INVALID_ARGUMENT"


def test_division_by_zero() -> None:
    assert err("math_eval", expr="1/0") == "MATH_ERROR"


def test_formatted_uses_english_separators() -> None:
    resp = call("math_eval", expr="1234567.891", precision=2)
    assert "1,234,567.89" in resp["formatted"]


def test_show_steps_fractions() -> None:
    r = ok("math_eval", expr="1/2 + 1/3", exact_mode="fraction", show_steps=True)
    assert r["value"] == "5/6"
    steps = r["steps"]
    assert [s["n"] for s in steps] == list(range(1, len(steps) + 1))
    assert len(steps) >= 3
    assert "check" in steps[-1]["text"].lower()
