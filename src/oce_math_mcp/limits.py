"""Anti-abuse limits (decisions D6, D16, D21 and spec V3, section 2.3)."""

from __future__ import annotations

EXPR_MAX_CHARS = 2_000
EXPONENT_MAX = 10_000
INT_BITS_MAX = 4_000_000
MATRIX_MAX = 50
LIST_MAX = 10_000
INT_DIGITS_MAX = 10_000
INT_SUMMARY_DIGITS = 10_000
DIGITS_MAX = 1_000
SANDBOX_CODE_MAX = 8_000
SANDBOX_OUTPUT_MAX = 10_000
SANDBOX_SEQ_MAX = 10_000_000
LEAN_CODE_MAX = 20_000
TIMEOUT_S = 5.0
LEAN_TIMEOUT_S = 60.0
# D104: degree of legendre/hermite/laguerre and of math_numeric.polyroots. Measured 2026-11-26: building
# laguerre(200, x) 0.30 s, laguerre(300, x) 1.06 s, laguerre(500, x) 6.07 s (over the 5 s limit per call).
POLY_DEGREE_MAX = 200
# D103: loading Mathlib into the Lean REPL took 16.16 s (measured 2026-11-26); a cold disk can take much longer.
LEAN_REPL_START_S = 300.0
# D103: +0.24 MB per check (1,000 checks: 3,130 -> 3,374 MB in use); the REPL restarts after this many checks.
LEAN_REPL_RECYCLE = 1000
MEMORY_MB = 2_048
