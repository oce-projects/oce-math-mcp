"""D119: eng_calc.special_relativity mode doppler with an angle (angle_deg, angle_frame).

References: page "Relativistic Doppler effect", section "Motion in an arbitrary direction":
f_r = f_s / (gamma (1 + beta cos theta_r)) with the angle in the frame of the receiver,
f_r = gamma (1 - beta cos theta_s) f_s with the angle in the frame of the source (Einstein 1905),
cos theta_r = (cos theta_s - beta) / (1 - beta cos theta_s) (relativistic aberration); beta > 0 when source and
receiver move away from each other, theta = 0 along the line of sight. Expected values in plain Python (math).
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from oce_math_mcp.envelope import Ctx
from oce_math_mcp.errors import ToolError
from oce_math_mcp.tools import eng_calc


def run(params: dict[str, Any], ctx: Ctx | None = None) -> dict[str, Any]:
    return eng_calc.handle(
        {"operation": "special_relativity", "params": {"mode": "doppler", **params}}, ctx or Ctx()
    )


def gamma(b: float) -> float:
    return 1 / math.sqrt(1 - b * b)


def test_the_measure_angle_in_the_source_frame() -> None:
    """RAG doc 62, Doppler shift 1 and 9: nu' = gamma nu (1 - beta cos theta), nu = 5e14 Hz, beta = 0.3, 1 rad."""
    out = run({"f": 5e14, "beta": 0.3, "angle_deg": math.degrees(1.0), "angle_frame": "source"})
    want = gamma(0.3) * 5e14 * (1 - 0.3 * math.cos(1.0))
    assert out["f_observed"] == pytest.approx(want, rel=1e-14, abs=0)
    assert out["angle_source_deg"] == pytest.approx(math.degrees(1.0), rel=1e-14, abs=0)
    assert out["gamma"] == pytest.approx(gamma(0.3), rel=1e-14, abs=0)  # outputs have 15 digits
    assert out["beta"] == 0.3


def test_transverse_in_the_source_frame_is_a_blueshift() -> None:
    """Doppler shift 3 and 11: nu' = gamma nu at theta_s = 90 degrees."""
    out = run({"f": 5e14, "beta": 0.3, "angle_deg": 90, "angle_frame": "source"})
    assert out["f_observed"] == pytest.approx(gamma(0.3) * 5e14, rel=1e-14, abs=0)
    assert out["z"] == pytest.approx(1 / gamma(0.3) - 1, rel=1e-13, abs=0)
    # aberration: cos theta_r = -beta
    assert out["angle_receiver_deg"] == pytest.approx(
        math.degrees(math.acos(-0.3)), rel=1e-14, abs=0
    )


def test_transverse_in_the_receiver_frame_is_a_redshift() -> None:
    out = run({"f": 5e14, "beta": 0.3, "angle_deg": 90, "angle_frame": "receiver"})
    assert out["f_observed"] == pytest.approx(5e14 / gamma(0.3), rel=1e-14, abs=0)
    assert out["z"] == pytest.approx(gamma(0.3) - 1, rel=1e-13, abs=0)
    assert out["receding"] is False  # no radial component in the frame of the receiver


def test_the_two_frames_agree_through_the_aberration() -> None:
    b, th_r = 0.6, math.radians(60)
    out_r = run({"f": 1e9, "beta": b, "angle_deg": 60, "angle_frame": "receiver"})
    want = 1e9 / (gamma(b) * (1 + b * math.cos(th_r)))
    assert out_r["f_observed"] == pytest.approx(want, rel=1e-14, abs=0)
    th_s = math.acos((math.cos(th_r) + b) / (1 + b * math.cos(th_r)))  # the inverse aberration
    assert out_r["angle_source_deg"] == pytest.approx(math.degrees(th_s), rel=1e-13, abs=0)
    assert out_r["angle_receiver_deg"] == 60
    out_s = run({"f": 1e9, "beta": b, "angle_deg": math.degrees(th_s), "angle_frame": "source"})
    assert out_s["f_observed"] == pytest.approx(want, rel=1e-13, abs=0)
    assert out_s["angle_receiver_deg"] == pytest.approx(60, rel=1e-13, abs=0)
    assert out_s["z"] == pytest.approx(out_r["z"], rel=1e-12, abs=0)


@pytest.mark.parametrize("frame", ["source", "receiver"])
def test_along_the_line_of_sight_the_result_is_the_old_one(frame: str) -> None:
    old = run({"f": 5e14, "beta": 0.3})
    away = run({"f": 5e14, "beta": 0.3, "angle_deg": 0, "angle_frame": frame})
    assert away["f_observed"] == pytest.approx(old["f_observed"], rel=1e-14, abs=0)
    assert away["z"] == pytest.approx(old["z"], rel=1e-14, abs=0)
    assert away["receding"] is True
    towards = run({"f": 5e14, "beta": 0.3, "angle_deg": 180, "angle_frame": frame})
    old_towards = run({"f": 5e14, "beta": -0.3})
    assert towards["f_observed"] == pytest.approx(old_towards["f_observed"], rel=1e-14, abs=0)
    assert towards["receding"] is False


def test_wavelength_and_a_negative_beta() -> None:
    b = -0.4  # approaching
    out = run({"wavelength": 5e-7, "beta": b, "angle_deg": 180, "angle_frame": "receiver"})
    z = gamma(b) * (1 + b * math.cos(math.pi)) - 1  # f_s / f_r - 1
    assert out["wavelength_observed"] == pytest.approx(5e-7 * (1 + z), rel=1e-14, abs=0)
    assert out["receding"] is True  # v cos theta_r = (-0.4)(-1) > 0: moving away


def test_small_beta_has_no_cancellation() -> None:
    """z = gamma - 1 = beta^2 / 2 + 3 beta^4 / 8 + ... at theta_r = 90 degrees."""
    b = 1e-9
    out = run({"f": 1e9, "beta": b, "angle_deg": 90, "angle_frame": "receiver"})
    assert out["z"] == pytest.approx(b * b / 2, rel=1e-12, abs=0)


def test_units() -> None:
    ctx = Ctx()
    run({"f": 1e9, "beta": 0.5, "angle_deg": 30, "angle_frame": "source"}, ctx)
    assert ctx.units is not None
    assert ctx.units["angle_source_deg"] == "deg"
    assert ctx.units["angle_receiver_deg"] == "deg"
    assert ctx.units["f_observed"] == "Hz"


@pytest.mark.parametrize(
    ("params", "words"),
    [
        ({"f": 1e9, "beta": 0.5, "angle_deg": 30}, "angle_frame"),
        ({"f": 1e9, "beta": 0.5, "angle_frame": "source"}, "needs params.angle_deg"),
        ({"f": 1e9, "beta": 0.5, "angle_deg": 30, "angle_frame": "observer"}, "angle_frame"),
        ({"f": 1e9, "beta": 0.5, "angle_deg": -1, "angle_frame": "source"}, "between 0 and 180"),
        ({"f": 1e9, "beta": 0.5, "angle_deg": 181, "angle_frame": "source"}, "between 0 and 180"),
        ({"z": 0.1, "angle_deg": 30, "angle_frame": "source"}, "angle"),
    ],
)
def test_errors(params: dict[str, Any], words: str) -> None:
    with pytest.raises(ToolError) as err:
        run(params)
    assert err.value.code == "INVALID_ARGUMENT"
    assert words in err.value.message
