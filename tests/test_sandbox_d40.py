"""D40 (approved on 2026-11-24): `vars` as a name, `time` and `datetime` modules in python_sandbox.

Cases taken from the OpenMathReasoning measurement of 2026-11-24 (FORBIDDEN: `vars` 112, `time` 31,
`datetime` 9); expected results from the dataset or from known calendar dates.
"""

from __future__ import annotations

import textwrap

import pytest

from tests.conftest import call, ok


def run(code: str) -> dict:
    return ok("python_sandbox", code=textwrap.dedent(code))


def code_of(code: str) -> str:
    r = call("python_sandbox", code=textwrap.dedent(code))
    assert r["ok"] is False, r
    return r["error"]["code"]


# ------------------------------------------------------------------ vars as a name


def test_vars_as_argument_fsolve_like_dataset() -> None:
    r = run(
        """
        from scipy.optimize import fsolve
        import numpy as np

        def equations(vars):
            x, y, z = vars
            eq1 = 23*x + 24*y + 25*z - (x**5 + y**5)
            eq2 = 23*y + 24*z + 25*x - (y**5 + z**5)
            eq3 = 23*z + 24*x + 25*y - (z**5 + x**5)
            return [eq1, eq2, eq3]

        solution = fsolve(equations, [1, 2, 3])
        [round(float(v), 8) for v in solution]
        """
    )
    assert (
        r["value"] == "[2.44948974, 2.44948974, 2.44948974]"
    )  # expected in the dataset (line 28243)


def test_vars_as_variable_and_lambda() -> None:
    assert run("vars = [1, 2, 3]\nsum(vars)")["value"] == "6"
    assert run("f = lambda vars: vars[0] + vars[1]\nf((4, 5))")["value"] == "9"
    assert run("[vars * 2 for vars in range(3)]")["value"] == "[0, 2, 4]"


@pytest.mark.parametrize("code", ["vars()", "import sympy as sp\nvars(sp)", "x = 1\nvars"])
def test_builtin_vars_still_forbidden(code: str) -> None:
    assert code_of(code) == "FORBIDDEN"


def test_builtin_vars_not_reachable_when_name_bound_elsewhere() -> None:
    r = call(
        "python_sandbox", code="def f(vars):\n    return vars\ndef g():\n    return vars()\ng()"
    )
    assert r["ok"] is False  # NameError: the vars builtin does not exist in the sandbox


# ------------------------------------------------------------------ time


def test_time_for_timing() -> None:
    r = run(
        """
        import time
        start = time.time()
        total = sum(range(10000))
        elapsed = time.time() - start
        (total, elapsed >= 0, time.perf_counter() > 0, time.monotonic() > 0)
        """
    )
    assert r["value"] == "(49995000, True, True, True)"


def test_time_star_import_and_struct() -> None:
    assert run("from time import *\nperf_counter() >= 0")["value"] == "True"
    assert run("import time\ntime.gmtime(0).tm_year")["value"] == "1970"


@pytest.mark.parametrize(
    "code",
    [
        "import time\ntime.sleep(1)",
        "from time import sleep",
        "import time\ntime.get_clock_info('time')",
    ],
)
def test_time_outside_whitelist_forbidden(code: str) -> None:
    assert code_of(code) == "FORBIDDEN"


# ------------------------------------------------------------------ datetime


def test_datetime_weekday_and_leap_february() -> None:
    r = run(
        """
        import datetime
        (datetime.date(2000, 1, 1).weekday(), (datetime.date(2024, 3, 1) - datetime.date(2024, 2, 1)).days)
        """
    )
    assert r["value"] == "(5, 29)"  # 2000-01-01 was a Saturday; 2024 is a leap year


def test_datetime_from_import_and_strftime() -> None:
    r = run(
        """
        from datetime import date, timedelta
        d = date(2026, 11, 24) + timedelta(days=7)
        (d.isoformat(), d.strftime('%d/%m/%Y'))
        """
    )
    assert r["value"] == "('2026-12-01', '01/12/2026')"


@pytest.mark.parametrize(
    "code",
    [
        "import datetime\ndatetime.datetime_CAPI",
        "import datetime\ndatetime.sys",
        "import datetime\ndatetime.date.__reduce__",
    ],
)
def test_datetime_outside_whitelist_forbidden(code: str) -> None:
    assert code_of(code) == "FORBIDDEN"
