"""D100: eng_calc.waves (modulation indices and general wave functions) and the wave equations checked by
math_verify.check_solution.

References: direct sums of the sine waves written here, mpmath integrals and roots, the FM broadcast textbook
value (Carson bandwidth 180 kHz for 75 kHz deviation and 15 kHz audio).
"""

from __future__ import annotations

import cmath
import math
from typing import Any

import mpmath
import pytest

from oce_math_mcp.envelope import Ctx
from oce_math_mcp.errors import ToolError
from oce_math_mcp.tools import eng_calc
from tests.conftest import call

QUAD: Any = mpmath.quad  # the mpmath stubs type quad as a tuple


def run(params: dict, ctx: Ctx | None = None) -> dict:
    return eng_calc.handle({"operation": "waves", "params": params}, ctx or Ctx())


def approx(v: Any, rel: float = 1e-13) -> Any:
    return pytest.approx(v, rel=rel, abs=0)


def test_modulation_indices() -> None:
    out = run({"mode": "modulation", "kind": "AM", "A_mod": 0.5, "A_carrier": 2.0})
    assert out["h"] == 0.25
    assert out["envelope_max"] == approx(2.5) and out["envelope_min"] == approx(1.5)
    # sideband power: mean square of the AM signal minus the carrier, over the mean square (mpmath average)
    h = 0.25
    with mpmath.workdps(30):
        total = QUAD(
            lambda t: ((1 + h * mpmath.cos(t)) * mpmath.cos(40 * t)) ** 2, [0, 2 * mpmath.pi]
        ) / (2 * mpmath.pi)
    assert out["sideband_power_fraction"] == approx(float((total - 0.5) / total), rel=1e-12)
    ctx = Ctx()
    run({"mode": "modulation", "kind": "AM", "A_mod": 3, "A_carrier": 2}, ctx)
    assert any("over-modulation" in w for w in ctx.warnings)
    out = run({"mode": "modulation", "kind": "FM", "delta_f": 75e3, "f_m": 15e3})
    assert out["h"] == 5.0 and out["carson_bandwidth"] == approx(180e3)
    out = run({"mode": "modulation", "kind": "PM", "delta_phi": 0.8, "f_m": 1e3})
    assert out["h"] == 0.8 and out["delta_f"] == approx(800.0)
    with pytest.raises(ToolError):
        run({"mode": "modulation", "kind": "FM", "delta_f": 1e3, "f_m": 1e2, "A_mod": 1})


def test_damped_wave() -> None:
    out = run(
        {
            "mode": "damped",
            "A0": 2.0,
            "b": 0.5,
            "k": 3.0,
            "omega": 4.0,
            "phi": 0.3,
            "x": 1.1,
            "t": 0.7,
        }
    )
    want = 2.0 * math.exp(-0.35) * math.sin(3.3 - 2.8 + 0.3)
    assert out["A"] == approx(want) and out["tau"] == 2.0
    assert out["half_life"] == approx(2 * math.log(2))


def test_superposition_against_complex_sum() -> None:
    waves = [
        [1.0, [2.0, 0.0, 1.0], 3.0, 0.0],
        [0.5, [2.1, 0.3, 0.0], 3.2, 1.0],
        [-0.7, [0.0, 1.0, 1.0], 2.5, 0.2],
    ]
    r, t = [0.4, -0.2, 1.3], 0.9
    out = run({"mode": "superposition", "waves": waves, "point": r, "t": t})
    total = sum(
        a * cmath.exp(1j * (sum(ki * xi for ki, xi in zip(k, r, strict=True)) - w * t + ph))
        for a, k, w, ph in waves
    )
    assert out["re"] == approx(total.real) and out["im"] == approx(total.imag)
    assert out["amplitude"] == approx(abs(total))
    assert out["amplitude_pairs"] == approx(abs(total), rel=1e-12)  # formula 8 of the page
    assert out["phase"] == approx(cmath.phase(total))
    with pytest.raises(ToolError):
        run({"mode": "superposition", "waves": [[1, [1, 2], 3, 0]], "point": 0.5})
    with pytest.raises(ToolError, match="1000"):
        run({"mode": "superposition", "waves": [[1, 1, 1, 0]] * 1001, "point": 0.0})


def test_standing_wave_nodes_and_value() -> None:
    a, k, w, x, t = 0.3, 2 * math.pi, 5.0, 0.37, 0.11
    out = run({"mode": "standing", "A": a, "k": k, "omega": w, "x": x, "t": t, "L": 2.0})
    direct = a * math.sin(k * x - w * t) + a * math.sin(k * x + w * t)
    assert out["y"] == approx(direct, rel=1e-12)
    assert out["nodes"] == approx([0.0, 0.5, 1.0, 1.5, 2.0])
    assert out["antinodes"] == approx([0.25, 0.75, 1.25, 1.75])
    assert out["k"] == approx(k) and out["omega"] == approx(w)
    assert out["wavelength"] == approx(1.0)
    with pytest.raises(ToolError, match="nodes"):
        run({"mode": "standing", "A": a, "k": k, "omega": w, "L": 6000.0})  # 12.001 nodes
    same = run(
        {"mode": "standing", "A": a, "wavelength": 1.0, "f": w / (2 * math.pi), "x": x, "t": t}
    )
    assert same["y"] == approx(out["y"], rel=1e-12)


def test_beats() -> None:
    w1, w2 = 2 * math.pi * 444, 2 * math.pi * 440
    k1, k2 = w1 / 343, w2 / 343
    out = run(
        {
            "mode": "beats",
            "A": 0.2,
            "k1": k1,
            "omega1": w1,
            "k2": k2,
            "omega2": w2,
            "x": 1.7,
            "t": 0.013,
        }
    )
    assert out["f_beat"] == approx(4.0, rel=1e-12) and out["T_beat"] == approx(0.25, rel=1e-12)
    assert out["y"] == approx(out["y_direct"], rel=1e-10)
    assert out["v_phase"] == approx(343.0) and out["v_group"] == approx(343.0, rel=1e-10)
    assert out["omega_mean"] == approx((w1 + w2) / 2) and out["delta_k"] == approx(k1 - k2)


@pytest.mark.parametrize("phi", [0.0, 1.0, 2.5, 4.0])
def test_phase_shift(phi: float) -> None:
    out = run(
        {"mode": "phase_shift", "A": 1.5, "phi": phi, "k": 2.0, "omega": 3.0, "x": 0.4, "t": 0.1}
    )
    assert out["y"] == approx(out["y_direct"], rel=1e-12)
    # amplitude and phase reproduce the direct sum at any point
    arg = 2.0 * 0.4 - 3.0 * 0.1
    assert out["amplitude"] * math.sin(arg + out["phase"]) == approx(out["y_direct"], rel=1e-12)
    assert out["amplitude"] >= 0


def test_kdv_soliton_profile_and_equation() -> None:
    v, alpha = 1.7, 6.0
    out = run({"mode": "soliton", "v": v, "alpha": alpha, "x": 0.8, "t": 0.2})
    assert out["peak"] == approx(3 * v / alpha)
    with pytest.raises(ToolError, match="alpha"):
        run({"mode": "soliton", "v": v, "alpha": 0})
    assert out["y"] == approx(3 * v / alpha / math.cosh(math.sqrt(v) / 2 * (0.8 - v * 0.2)) ** 2)

    def prof(x: Any) -> Any:
        return 3 * v / alpha / mpmath.cosh(mpmath.sqrt(v) / 2 * x) ** 2

    with mpmath.workdps(30):
        area = QUAD(prof, [-mpmath.inf, mpmath.inf])
        half = mpmath.findroot(lambda x: prof(x) - out["peak"] / 2, 1.0)
    assert out["area"] == approx(float(area), rel=1e-12)
    assert out["fwhm"] == approx(float(2 * half), rel=1e-12)
    # the profile solves the KdV equation of the page (symbolic check by math_verify)
    r = call(
        "math_verify",
        operation="check_solution",
        expr="diff(y, t) + a*y*diff(y, x) + diff(y, x, 3) = 0",
        solution={"y": "3*c/a/cosh(sqrt(c)/2*(x - c*t))**2"},
    )
    assert r["ok"] and r["result"]["valid"] is True


def test_wave_equation_by_check_solution() -> None:
    eq = "diff(A, x, 2) + diff(A, y, 2) + diff(A, z, 2) - diff(A, t, 2)/v**2 = 0"
    good = call(
        "math_verify", operation="check_solution", expr=eq, solution={"A": "exp(-(x - v*t)**2)"}
    )
    assert good["ok"] and good["result"]["valid"] is True
    bad = call(
        "math_verify", operation="check_solution", expr=eq, solution={"A": "exp(-(x - 2*v*t)**2)"}
    )
    assert bad["ok"] and bad["result"]["valid"] is False
