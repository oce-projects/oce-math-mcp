# oce-math-mcp benchmark

File written by `oce-math-mcp --benchmark --benchmark-md docs/BENCHMARK.md`: the numbers come
from the program, they are not written by hand.

- Version 0.3.0, measured on 2026-11-24 20:06 UTC.
- Python 3.13.13, Windows-10-10.0.19045-SP0, 12 logical CPUs.

| Measurement | Value |
|---|---|
| Start of one computing process (median of 3) | 3.603 s |
| Single starts | 3.650 s; 3.603 s; 3.477 s |
| Warm `math_eval` call in the pool (median, p95; 50 calls) | 0.24 ms; 0.34 ms |
| `math_eval` call in the same process (median, p95; 50 calls) | 0.09 ms; 0.10 ms |
| Response to an endless loop with a 1 s timeout | 1.011 s (code TIMEOUT) |
| First call after the kill (pool of 1 process) | 3.514 s |
| First call after `python_sandbox` (pool of 1) | 3.349 s |
| `lean_check` of `2 + 2 = 4` with Mathlib | 16.619 s |
