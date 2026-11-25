"""Two caps of python_sandbox that the audit of 25/11/2026 could pass (D94).

`[0] * np.int64(20000000)` built 20,000,000 elements (cap 10,000,000): numpy integers were not seen as
integers by the guard. `(1 << 3999999) << 3999999` built 7,999,999 bits (cap 4,000,000): only the shift was
checked, not the size of the result.
"""

from __future__ import annotations

from tests.conftest import call, ok


def run(code: str) -> dict:
    return call("python_sandbox", code=code)


def test_sequence_cap_with_numpy_integers() -> None:
    for code in (
        "import numpy as np\nx = [0] * np.int64(20000000)\nlen(x)",
        "import numpy as np\nx = 'ab' * np.int32(20000000)\nlen(x)",
        "import numpy as np\nx = np.int64(20000000) * (1,)\nlen(x)",
    ):
        resp = run(code)
        assert not resp["ok"] and resp["error"]["code"] == "LIMIT_EXCEEDED", (code, resp)
    assert ok("python_sandbox", code="import numpy as np\nlen([0] * np.int64(5))")["value"] == "5"
    assert (
        ok("python_sandbox", code="import numpy as np\nnp.int64(3) * 'ab'")["value"] == "'ababab'"
    )


def test_shift_cap_is_on_the_result() -> None:
    for code in (
        "x = 1 << 3999999\ny = x << 3999999\ny.bit_length()",
        "x = 1 << 3999999\nx <<= 2\nx.bit_length()",
        "x = 3 << 3999999\nx.bit_length()",
        "x = 1 << 3999999\ny = x << 1\ny.bit_length()",
    ):
        resp = run(code)
        assert not resp["ok"] and resp["error"]["code"] == "LIMIT_EXCEEDED", (code, resp)
    assert ok("python_sandbox", code="(1 << 3999999).bit_length()")["value"] == "4000000"
    assert ok("python_sandbox", code="((1 << 100) << 100).bit_length()")["value"] == "201"
    assert ok("python_sandbox", code="(5 << -0) + (8 >> 1)")["value"] == "9"
