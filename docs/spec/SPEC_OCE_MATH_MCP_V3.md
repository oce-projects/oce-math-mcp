# `oce-math-mcp` V3 specification: final tool catalogue

Spec version: **4.13.0** of 2026-11-25 (3.1.0: 3D drawing operations in `geom3d_calc` and the
`show_steps` field, decisions D22 and D23; 3.2.0: `date_calc` with recurring events, time zones, real
hours between two instants, age in months and weeks, decision D24; 3.3.0: tool status in server
0.1.0, `lean_check` with code filter and axiom check, decision D36; 3.4.0:
`python_sandbox` with `from ... import *`, `global`, `sympy.S` and `nsimplify`, decision D38; JSON
schemas unchanged; 3.5.0: `eng_calc` area A with the `params` keys, decision D39; 3.6.0: `eng_calc` area B;
3.7.0: `eng_calc` areas C and D, JSON schemas unchanged;
3.8.0: field `events` of `math_numeric.ode_ivp`, decision D41;
3.9.0: `math_verify.check_units` and field `units`, decision D42;
3.10.0: `python_sandbox` allows `vars` as a name and part of the `time` and `datetime` modules, decision
D40; JSON schemas unchanged;
3.11.0: `eng_calc` areas E and F, 8 new operations, decision D43;
3.12.0: the 4 `eng_calc` operations with data tables, decision D44; JSON schemas unchanged;
3.13.0: 10 new modes in `kinematics_dynamics`, `optics_acoustics`, `circuit_ac_dc`, decision D45; JSON schemas unchanged;
3.14.0: 5 new `eng_calc` operations, strength of materials without tables, decision D46;
3.15.0: CODATA physical constants, function `const()` in `math_eval` and `eng_calc.physical_constant`,
decision D47;
3.16.0: 6 chemistry operations in `eng_calc`, decision D48;
3.17.0: 5 operations for nuclear and modern physics and relativity in `eng_calc`, decision D49;
3.18.0: IAPWS-IF97 water and steam in `eng_calc.water_steam`, decision D50;
3.19.0: psychrometrics in `eng_calc.psychrometrics`, decision D51;
3.20.0: Nusselt correlations in `eng_calc.convection_correlation`, decision D52;
3.21.0: e-NTU heat exchangers, fins and lumped-parameter transients, decision D53;
3.22.0: radiation between surfaces in `eng_calc.radiation_exchange`, decision D54;
3.23.0: electrostatics and capacitance in `eng_calc.electrostatics`, decision D55;
3.24.0: magnetism, inductance and electromagnetic induction, decision D56;
3.25.0: linkages and epicyclic gear trains, decision D57;
3.26.0: catenary cables and journal bearings, decision D58;
3.27.0: 2D and 3D trusses, decision D59;
3.28.0: torsion of non-circular sections, decision D60;
4.0.0: everything in English, decision D61 (breaking: `formatted_it` -> `formatted`, step `text_it` -> `text`, English separators and ISO dates in texts, `vat`/`vat_add`/`vat_remove`/`withholding` in `math_eval`, no `weekday_it` in `date_calc`, English text values in `eng_calc`, no Italian particle aliases); 4.1.0: geometric and wave optics, decision D64; 4.2.0: practical electronics, decision D65; 4.3.0: friction in simple machines, decision D66; 4.4.0: electric machines, decision D67; 4.5.0: thin plates, decision D68; 4.6.0: continuous beams, decision D69; 4.7.0: local losses, decision D70; 4.8.0: stress concentration at holes, decision D71; 4.9.0: rotor balancing, decision D72; 4.10.0: rolling without slipping, decision D73; 4.10.1: finite results in eng_calc, decision D74; 4.11.0: kinetic theory of gases, decision D87; 4.12.0: Newtonian gravitation, decision D88; 4.12.1: `gravitation.rotating_body` towards the equator in both hemispheres, decision D89; 4.13.0: corrections of the audit of the repository, decisions D90-D97). Server: `oce-math-mcp` (package `oce_math_mcp`).

**This spec replaces V2** (`SPEC_OCE_MATH_MCP_E_PIANO_V2.md`) for the tool catalogue and for the
JSON schemas. The decisions behind it are D13-D21 in `docs/DECISIONS.md`.

Identical copies, to be updated at the same time:

- repo: `docs/spec/SPEC_OCE_MATH_MCP_V3.md` and `docs/spec/oce_math_mcp_tools.json`
- dataset: `SPEC_OCE_MATH_MCP_V3.md` and `oce_math_mcp_tools.json` in the dataset folder (`OCE_MATH_DATASET_DIR`)
- knowledge base: the spec document of the project knowledge base (`OCE_MATH_RAG_DOC`)

The file `oce_math_mcp_tools.json` is the **machine contract**. The command
`oce-math-mcp --dump-schemas` prints the same content. A test checks that the file in the repo,
the JSON block at the end of this spec and the schemas exposed by the server are equal.

---

## 1. Catalogue

| No. | Tool | What it is for | Status |
|---:|---|---|---|
| 1 | `math_eval` | exact arithmetic, functions, statistics, Italian finance, units | done in 0.1.0 |
| 2 | `math_solve` | symbolic SymPy with self-check: Calculus 1 and 2, ODE, transforms, linear algebra, summations, recurrences, polynomials | done in 0.1.0 |
| 3 | `number_theory` | primes, factorisation, modular arithmetic, CRT, Diophantine equations, continued fractions | done in 0.1.0 |
| 4 | `math_numeric` | arbitrary precision (mpmath), integrals, roots, optimisation, LP/ILP, numerical ODE, fit | done in 0.1.0 |
| 5 | `math_verify` | check of an answer before giving it | done in 0.1.0 |
| 6 | `python_sandbox` | short programs for problems without a dedicated tool (counting, brute force, sequences) | done in 0.1.0 |
| 7 | `lean_check` | formal proofs in Lean 4 + Mathlib | done in 0.1.0 |
| 8 | `date_calc` | Italian calendar `Europe/Rome`, age, holidays, daylight saving time, deadlines | done in 0.1.0 |
| 9 | `geom3d_calc` | 3D geometry, CAD, CNC, robotics, calculations for 3D drawing | done in 0.1.0 |
| 10 | `eng_calc` | engineering formulas (25 operations) | **21 of 25 in 0.2.0 (areas A, B, C, D); the 4 with data tables in phase 2** |

The `eng_calc` operations not yet written answer `ok: false` with error `NOT_IMPLEMENTED`.
The `eng_calc` examples for `mix_v4` are generated only for the ready operations (D12: the answers
come from the real server).

Rule for Qwen: **the dedicated tool first; `python_sandbox` only if no tool has the requested
operation; `math_verify` before giving an important final answer.**

---

## 2. Common rules

### 2.1 Shape of the response

Every tool answers with a JSON object with **9 fields, always present**, in the same order:

```json
{
  "ok": true,
  "tool": "math_solve",
  "operation": "integrate",
  "result": {"integral": "x**3/3", "definite": false, "numeric": null, "verified": true},
  "units": null,
  "formula_used": null,
  "formatted": "∫ x**2 dx = x**3/3 + C",
  "warnings": [],
  "error": null
}
```

| Field | Type | Meaning |
|---|---|---|
| `ok` | boolean | `true` if the calculation succeeded |
| `tool` | string | name of the tool |
| `operation` | string or `null` | operation performed; `null` for `math_eval`, `python_sandbox`, `lean_check` |
| `result` | object or `null` | result; `null` if `ok` is `false` |
| `units` | object or `null` | units of the values in `result`, when meaningful |
| `formula_used` | string or `null` | formula applied, when there is only one |
| `formatted` | string | sentence in English, numbers in English format (`1,234.56`), dates in ISO format |
| `warnings` | list of strings | warnings in English (rounding, conventions, truncated results) |
| `error` | object or `null` | error, see 2.2 |

The exact values in `result` are **strings in SymPy syntax** (`"sqrt(2)/2"`, `"pi/4"`, `"1/3"`),
so they can be passed on to another tool. Values that are only numeric are JSON numbers.

The internal keys of `result` are described in the tool sections. The **input schemas** are
the contract "to the byte"; the keys of `result` are stable from server version 1.0.0.

### 2.2 Errors

```json
{"code": "PARSE_ERROR", "message": "Implicit multiplication not allowed: '2x'.", "example": "2*x + 1"}
```

| Code | When |
|---|---|
| `INVALID_ARGUMENT` | missing field, wrong type, value outside the schema domain |
| `PARSE_ERROR` | unreadable expression |
| `FORBIDDEN` | construct forbidden by security (attributes, `__`, imports not allowed) |
| `LIMIT_EXCEEDED` | a limit of table 2.3 exceeded |
| `TIMEOUT` | calculation over the maximum time; the calculation process is killed |
| `MATH_ERROR` | division by zero, function domain, no convergence |
| `NOT_IMPLEMENTED` | operation planned but not yet written (4 `eng_calc` operations) |
| `UNAVAILABLE` | external component missing (Lean not installed) |
| `INTERNAL_ERROR` | unexpected error; the message contains no traceback or paths |

`example` contains, when possible, a correct value for the same field.

### 2.3 Limits (D6, D16, D21)

| Limit | Value |
|---|---|
| Length of `expr` and of the other expression fields | 2,000 characters |
| Numeric exponent | absolute value <= 10,000 |
| Matrices | <= 50 rows and <= 50 columns |
| Elements in an input list | <= 10,000 |
| Digits of an input integer | <= 10,000 |
| `math_numeric.digits` | <= 1,000 |
| `python_sandbox.code` | <= 8,000 characters; output <= 10,000 characters, beyond that it is truncated with a warning |
| `lean_check.code` | <= 20,000 characters |
| Time per call | 5.0 s; `lean_check` 60 s |
| Memory of the calculation process | 2 GiB (Windows: Job Object; Linux: `RLIMIT_AS`) |

### 2.4 Security (D5, D13)

- No `eval`, `exec`, `compile`, `sympify`, `parse_expr`, `__import__` on the input text.
- Expressions go through `ast.parse` and an allow-list builder that creates the SymPy objects
  directly. Allowed nodes: numeric constants, names, arithmetic operators, comparisons, calls to
  functions of the tool list, lists and tuples as arguments. Everything else is `FORBIDDEN`.
- Every calculation runs in an already started **separate process**. Over the maximum time the process is
  killed and replaced.
- In the calculation process the SymPy string-to-expression conversion (`parse_expr`) is replaced by
  a version that accepts only names and numbers: no string ever reaches `eval`.

### 2.5 Expression syntax

| Element | Rule |
|---|---|
| Numbers | `3`, `-2.5`, `1e-3`; decimals are **exact** (`0.1` = 1/10). The Italian decimal comma is not allowed (`3,5` → `PARSE_ERROR` with example `3.5`) |
| Power | `**` or `^` |
| Operators | `+ - * / // %`, round brackets |
| Multiplication | always explicit: `2*x`, not `2x` |
| Equation | a single `=` (`==` is accepted as `=`) |
| Inequality | `<`, `<=`, `>`, `>=` |
| Several equations or components | separated by `;` |
| Constants | `pi`, `E` (or `e`), `I` (imaginary unit), `oo` (infinity), `tau`, `GoldenRatio` (or `phi`). `e` and `phi` are variables if they appear in `variables` |
| Derivatives in ODEs | `y'`, `y''`, `y'''` or `diff(y(x), x, 2)` |
| Names | letters, digits, `_`; `__` is forbidden |

### 2.6 Output numbers

- `math_eval` in `decimal` mode rounds with `ROUND_HALF_UP` (commercial rounding) to
  `precision` decimal digits and removes useless trailing zeros.
- `formatted` uses the English format: comma for thousands, point for decimals; dates are ISO (`2026-11-24`).
- Integers with more than 10,000 digits: `result` gives the number of digits, the first 50 and the last 50 digits, with a warning.

### 2.7 Steps (`show_steps`, D23)

`math_eval`, `math_solve`, `number_theory` and `geom3d_calc` have the field `show_steps` (default
`false`). With `true`, `result.steps` contains the numbered steps, like a proof written at
school: data, rule or formula applied, intermediate calculations, result, **reverse check**.

```json
"steps": [
  {"n": 1, "text": "Coefficients: a = 1, b = -5, c = 6.", "expr": null},
  {"n": 2, "text": "Discriminant: b^2 - 4ac.", "expr": "1"},
  {"n": 3, "text": "Quadratic formula: x = (-b ± sqrt(Δ)) / (2a).", "expr": "[2, 3]"},
  {"n": 4, "text": "Check: substituting x = 2 and x = 3 the equation gives 0.", "expr": "[0, 0]"}
]
```

Detailed steps for: first and second degree equations, inequalities, linear systems,
derivatives (rule used at every step), integrals (rules of `sympy.integrals.manualintegrate`:
substitution, by parts, partial fractions; definite ones with antiderivative and values at the endpoints), limits
(direct substitution, indeterminate forms, l'Hôpital), arithmetic (fractions, powers,
percentages, VAT), number theory (extended Euclidean algorithm, CRT, modular powers by successive
squaring) and geometry formulas. The other operations give short steps: data, method,
result, check, with a warning.

The steps are generated by the server from the real calculation, never written by a model.

---

## 3. `math_eval`

Evaluates **a scalar expression**. `variables` assigns numeric values (numbers or numeric strings).

Modes: `decimal` (default, exact and then rounded to `precision` digits), `fraction` (exact fraction
`p/q`; if the result is irrational: decimal value and a warning), `float` (floating point).

`result`: `value` (string), `exact` (SymPy string), `float` (number or `null`), `is_exact` (boolean),
`mode` (string).

Allowed functions:

| Group | Functions |
|---|---|
| Basic | `abs`, `floor`, `ceil`, `round(x, n)` (half up), `min`, `max`, `sum`, `prod`, `sign` |
| Powers and roots | `sqrt`, `cbrt`, `root(x, n)`, `exp` |
| Logarithms | `log(x)` natural, `log(x, b)` base b, `ln`, `log10`, `log2` |
| Trigonometry (radians) | `sin`, `cos`, `tan`, `cot`, `sec`, `csc`, `asin`, `acos`, `atan`, `atan2(y, x)` |
| Trigonometry (degrees) | `sind`, `cosd`, `tand`, `asind`, `acosd`, `atand`, `deg(x)`, `rad(x)` |
| Hyperbolic | `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh` |
| Combinatorics | `factorial`, `comb(n, k)`, `perm(n, k)`, `gcd`, `lcm`, `mod_inv(a, m)`, `mod(a, m)` |
| Statistics (lists) | `mean`, `median`, `mode`, `stdev` (sample), `pstdev`, `variance`, `pvariance`, `cov(x, y)`, `pearson(x, y)` |
| Distributions | `norm_pdf(x, mu, sigma)`, `norm_cdf`, `norm_ppf(p, mu, sigma)`, `lognorm_pdf(x, mu, sigma)`, `lognorm_cdf`, `t_cdf(x, df)`, `t_ppf(p, df)`, `chi2_cdf(x, df)`, `chi2_ppf(p, df)`, `poisson_pmf(k, lam)`, `poisson_cdf`, `binom_pmf(k, n, p)`, `binom_cdf` |
| Finance | `pmt(rate, nper, pv)` positive instalment for a positive loan; `npv(rate, flows)` with `flows[0]` at time 0; `irr(flows)`; `dcf(rate, flows, g)` flows from time 1 plus Gordon terminal value with growth `g`; `vat(net, rate=22)`, `vat_add`, `vat_remove(gross, rate=22)`, `withholding(net, rate=20)` (default rates: 22 % Italian VAT, 20 % Italian withholding tax) |
| Units | `unit_convert(value, "from", "to")`: length, area, volume, mass, force, pressure (`Pa`, `kPa`, `MPa`, `bar`, `psi`, `atm`, `mmHg`), torque (`Nm`, `kgm`, `lbf_ft`), energy (`J`, `kJ`, `kWh`, `cal`, `kcal`, `BTU`), power (`W`, `kW`, `HP`, `CV`), temperature (`C`, `K`, `F`), angles, speed, flow rate. Exact defining factors (inch 0.0254 m, pound 0.45359237 kg) |
| Constants (D47) | `const("name")`: CODATA 2022 physical constant as an exact number (rational of the decimal notation), by CODATA name (`const("Planck constant")`, any letter case) or short symbol: `c`, `h`, `hbar`, `e`, `N_A`, `k_B`, `R`, `F`, `G`, `g_n`, `m_e`, `m_p`, `m_n`, `m_u`, `u`, `m_d`, `m_alpha`, `alpha`, `R_inf`, `a_0`, `r_e`, `lambda_C`, `E_h`, `epsilon_0`, `mu_0`, `Z_0`, `sigma`, `b_wien`, `c1`, `c2`, `mu_B`, `mu_N`, `Phi_0`, `K_J`, `R_K`, `atm`, `eV`, `V_m`, `m_e_MeV`, `m_p_MeV`, `m_n_MeV`, `u_MeV`. Values in SI units (the `_MeV` ones in MeV); uncertainty with `eng_calc.physical_constant` |

---

## 4. `math_solve`

Every operation that allows it returns `verified`: `true` only if the check succeeded
(substitution of the solution, derivative of the antiderivative, numerical comparison).

| Operation | Fields used | Keys of `result` |
|---|---|---|
| `solve` | `expr` (equation or expression = 0), `variables`, `domain` | `solutions`, `count`, `verified` |
| `solve_system` | `expr` with `;`, `variables`, `domain` | `solutions` (list of variable→value objects), `verified` |
| `inequality` | `expr`, `variables` | `solution_set`, `verified` |
| `diff` | `expr`, `variables` (one), `order` | `derivative`, `verified` |
| `partial_diff` | `expr`, `variables` (order of differentiation, e.g. `["x","y","x"]`) | `derivative`, `verified` |
| `integrate` | `expr`, `variables`, `bounds` `[a, b]` optional (`"-oo"`, `"oo"` allowed) | `integral`, `definite`, `numeric`, `verified` |
| `multi_integrate` | `expr`, `bounds` `[[x,a,b],[y,c,d],...]` (the first is the innermost), `coords` | `integral`, `numeric`, `verified` |
| `line_integral` | `expr` curve `"x=cos(t); y=sin(t); z=0"`, `variables` `["t"]`, `bounds` `[a, b]`, `vector_field` (3 components: work; 1 component: scalar field integral in ds) | `integral`, `numeric`, `verified` |
| `surface_integral` | `expr` surface `"x=...; y=...; z=..."`, `bounds` `[[u,a,b],[v,c,d]]`, `vector_field` (3: flux; 1: scalar field in dS) | `integral`, `numeric`, `verified` |
| `limit` | `expr`, `variables`, `bounds` `[point, dir]` with dir `"+"`, `"-"`, `"+-"` | `limit`, `verified` |
| `series` | `expr`, `variables`, `bounds` `[x0]` (default 0), `order` = order of the remainder | `series`, `polynomial`, `verified` |
| `function_study` | `expr`, `variables` | `domain`, `zeros`, `y_intercept`, `derivative`, `critical_points`, `inflection_points`, `asymptotes`, `limits` |
| `simplify`, `factor`, `expand`, `partial_fractions` | `expr`, `variables` | `expression`, `verified` |
| `gradient`, `laplacian` | `expr`, `variables` | `gradient` / `laplacian` |
| `divergence`, `curl`, `jacobian` | `vector_field`, `variables` | `divergence` / `curl` / `jacobian` and `determinant` |
| `hessian` | `expr`, `variables` | `hessian`, `critical_points` with type `minimum`, `maximum`, `saddle`, `degenerate` |
| `lagrange_optimize` | `expr` objective, `constraints` (equalities; inequalities with KKT, at most 4), `variables` | `candidates`, `maximum`, `minimum` |
| `ode` | `expr` (e.g. `"y'' + y = 0"`), `variables` `["y","x"]` (dependent, then independent), `ics`, `bounds` `[x0, x1]` for the numerical case | `solution`, `method` (`symbolic` or `numeric`), `verified`; numerical: `x`, `y` |
| `laplace` | `expr` f(t), `variables` `["t","s"]` | `transform`, `verified` |
| `inv_laplace` | `expr` F(s), `variables` `["s","t"]` | `transform`, `verified` |
| `fourier_fft` | `numeric_samples` (FFT; `bounds` `[sampling_frequency]` optional) or `expr`, `variables`, `bounds` `[-L, L]`, `order` = number of harmonics (symbolic series) | FFT: `frequencies`, `magnitudes`, `phases`, `dominant_frequency`; series: `a0`, `an`, `bn`, `series` |
| `riccati_lqr` | `matrix_a`, `matrix_b`, `matrix_q`, `matrix_r`, `time` | `P`, `K`, `closed_loop_eigenvalues`, `residual_norm`, `verified` |
| `matrix` | `matrix_a`, `matrix_b` (for `solve`, `multiply`, `projection`), `matrix_op`; for `norm_p` the norm goes in `expr` (`"1"`, `"2"`, `"oo"`, `"fro"`) | depends on `matrix_op`: `value`, `matrix`, `pivots`, `L`/`U`/`P`, `Q`/`R`, `U`/`S`/`Vt`, `eigenvalues`, `eigenvectors`, `solution`; always `verified` |
| `optimize_num` | as `math_numeric.minimize`; `numeric_samples` = starting point | `x`, `fun`, `success`, `message` |
| `complex_ops` | `expr` with `I` | `rectangular`, `re`, `im`, `modulus`, `argument_rad`, `argument_deg`, `conjugate`, `polar` |
| `summation` | `expr` term, `variables` `["n"]`, `bounds` `[a, b]` (`b` can be `"oo"`) | `sum`, `numeric`, `verified` |
| `recurrence` | `expr` (e.g. `"a(n) = a(n-1) + a(n-2)"`), `variables` `["a","n"]`, `ics` | `solution`, `first_terms`, `verified` |
| `polynomial` | `expr` (two polynomials with `;` for `resultant` and `gcd`), `variables`, `poly_op` | `poly_op`, `value`, `verified` |

Coordinates of `multi_integrate` (ISO 80000-2): `polar` and `cylindrical` add the factor `r`;
`spherical` adds `r**2*sin(theta)` with `theta` the polar angle from +z and `phi` the azimuth. The names `r`,
`theta`, `phi` must appear in `bounds`.

Matrices: exact calculation with SymPy if all elements are exact and the matrix has at most 10
rows; otherwise floating-point NumPy, with a warning.

---

## 5. `number_theory`

Exact integers of any size within the limit of 10,000 digits. Large integers can be
passed as a string (`"123456789012345678901234567890"`).

| Operation | Fields | Keys of `result` |
|---|---|---|
| `is_prime` | `n` | `is_prime` |
| `factorint` | `n` | `factors` (prime→exponent, string keys), `factorization` |
| `divisors` | `n`, `limit` | `divisors`, `count`, `sum`, `truncated` |
| `totient`, `mobius` | `n` | `value` |
| `next_prime`, `prev_prime`, `nth_prime`, `prime_pi` | `n` | `value` |
| `primerange` | `n` (start), `m` (end, excluded), `limit` | `primes`, `count`, `truncated` |
| `gcd`, `lcm` | `values` or `n`, `m` | `value` |
| `gcd_ext` | `n`, `m` | `gcd`, `x`, `y` (Bézout: `n*x + m*y = gcd`) |
| `mod_pow` | `n` base, `k` exponent, `m` modulus | `value` |
| `mod_inv` | `n`, `m` | `value` |
| `sqrt_mod` | `n`, `m` | `roots` |
| `discrete_log` | `n` value, `k` base, `m` modulus | `value` |
| `primitive_root`, `multiplicative_order` | `n` (order: `n`, `m`) | `value` |
| `legendre`, `jacobi` | `n`, `m` | `value` |
| `crt` | `values` (remainders), `moduli` | `value`, `modulus` |
| `diophantine` | `expr`, `variables` | `solutions` (parametric forms) |
| `continued_fraction` | `expr` or `n`, `m` (`n/m`), `limit` | `terms`, `periodic`, `convergents` |
| `partitions` | `n` | `value` |
| `digits` | `n`, `k` base (default 10) | `digits`, `digit_sum`, `count` |

---

## 6. `math_numeric`

| Operation | Fields | Keys of `result` |
|---|---|---|
| `evalf` | `expr`, `digits` | `value` (string with `digits` significant digits) |
| `nsum` | `expr`, `variables` `["n"]`, `bounds` `[a, b]` (`"oo"` allowed), `digits` | `value` |
| `quad` | `expr`, `variables`, `bounds` `[a, b]` or `[[x,a,b],[y,c,d]]`, `digits` | `value`, `error_estimate` |
| `root` | `expr` (equations with `;`), `variables`, `x0` or `bounds` `[lo, hi]` for `brentq` | `x`, `residual` |
| `minimize` | `expr`, `variables`, `x0`, `bounds`, `constraints`, `method`, `direction` | `x`, `fun`, `success`, `message` |
| `linprog` | `expr` linear objective, linear `constraints`, `variables`, `bounds`, `direction`, `integer` | `x`, `fun`, `success`, `status` |
| `ode_ivp` | `expr` right-hand sides `"y2; -y1"`, `variables` `["t","y1","y2"]`, `x0` initial state, `bounds` `[t0, t1]`, `method`, `samples`, `events` (section 6.1) | `t`, `y`, `final`; with `events` also `t_end`, `t_events`, `y_events`, `stopped_by` |
| `polyroots` | `expr`, `variables`, `digits` | `roots` (complex strings) |
| `identify` | `expr` (numeric value), `digits` | `candidates` (closed forms), `nsimplify` |
| `curve_fit` | `expr` model, `variables` `["x"]` (the other letters are the parameters), `data_x`, `data_y`, `x0` | `params`, `r_squared`, `residuals` |

`method = "auto"`: `SLSQP` with constraints, `BFGS` without; `RK45` for `ode_ivp`. `ode_ivp` tolerances:
`rtol = 1e-10`, `atol = 1e-12`.

### 6.1 `ode_ivp` events (D41)

Field `events`: list of at most 10 objects `{"expr", "direction", "terminal"}`.

- `expr`: expression in the variables of `variables`; the event fires when it is zero.
- `direction`: `both` (default), `rising` (from negative to positive), `falling` (from positive to
  negative).
- `terminal`: `true` (default) stops the integration at the first zero; `false` only records the
  zeros.
- **A zero exactly at t0 does not count**: the sign just after t0 is obtained from the derivatives
  along the motion (up to the third). So a projectile that starts from the ground (`y = 0`) does not stop
  immediately, even with `direction: "both"`.

With `events` the result also has `t_end` (instant of the stop, or `t1`), `t_events` and
`y_events` (one list per event, at most 1,000 zeros each), `stopped_by` (index of the
event that stopped it, `null` if none). `t` and `y` are `samples` points on `[t0, t_end]`;
`final` is the state at `t_end`. If no terminal event fires before `t1` there is a warning.
Without `events` the result does not change.

Real example (projectile with quadratic air drag, non-terminal apex, stop at the
ground):

```json
{"operation": "ode_ivp", "expr": "vx; vy; -0.01*vx*sqrt(vx**2 + vy**2); -9.80665 - 0.01*vy*sqrt(vx**2 + vy**2)", "variables": ["t", "x", "y", "vx", "vy"], "x0": [0, 0, 21.2132, 21.2132], "bounds": [0, 20], "samples": 3, "events": [{"expr": "vy", "direction": "falling", "terminal": false}, {"expr": "y", "direction": "falling"}]}
```

Result:

```json
{"t": [0.0, 1.8621391019813303, 3.7242782039626605], "y": [[0.0, 32.16552384703345, 55.79402462604384], [0.0, 17.0638469353124, 3.1086244689504383e-15], [21.2132, 14.57969463085615, 10.777223406224802], [21.2132, -1.009460177322496, -16.37244033427797]], "final": [55.79402462604384, 3.1086244689504383e-15, 10.777223406224802, -16.37244033427797], "method": "RK45", "nfev": 314, "t_end": 3.7242782039626605, "t_events": [[1.7584176223552397], [3.7242782039626605]], "y_events": [[[30.641731688073737, 17.116330511736557, 14.803738895279142, 5.551115123125783e-16]], [[55.79402462604384, 3.1086244689504383e-15, 10.777223406224802, -16.37244033427797]]], "stopped_by": 1}
```

---

## 7. `math_verify`

| Operation | Fields | Keys of `result` |
|---|---|---|
| `equivalent` | `expr`, `expected`, `variables`, `samples`, `tolerance`, `domain` | `equivalent`, `method` (`symbolic` or `numeric`), `counterexample` |
| `check_solution` | `expr` (equations with `;`), `solution` | `valid`, `residuals` |
| `check_derivative` | `expr` f, `expected` f', `variables` | `valid`, `difference` |
| `check_antiderivative` | `expr` F, `expected` f, `variables` | `valid`, `difference` |
| `check_numeric` | `expr` (exact expression), `expected` (stated value), `tolerance` (relative) | `valid`, `exact_value`, `relative_error` |
| `check_inequality` | `expr` inequality, `variables`, `samples`, `domain` | `holds_on_samples`, `counterexample` |
| `check_units` | `expr` formula or equation, `units` (unit of each variable), `expected` optional (section 7.1) | `consistent`, `dimension`, `si_unit`, `scale_to_si`, `problems`, `scale_ratio`, `expected_ok` |

`equivalent` with `method = "numeric"` and `check_inequality` **are not proofs**: they say that
no counterexample was found on `samples` random points (fixed seed, repeatable result).

### 7.1 Unit check: `check_units` (D42)

Qwen uses it when it writes a formula of its own (outside `eng_calc`): it says whether the dimensions are consistent.

- `expr`: formula or equation (`=`, also `<`, `<=`); the two sides must have the same
  dimension.
- `units`: unit of **every** variable, for example `{"rho": "kg/m^3", "Cd": "1"}`. Units of
  `sympy.physics.units` looked up by name (274 on 2026-11-24), plus the SI prefixes (`kN`, `MPa`,
  `mA`, `uF`), attached exponents (`m2`, `kg/m3`) and the aliases `min`, `rpm`, `rev`, `degC`,
  `degF`, `Nm`, `cP`, `cSt`, `Wh`, `Ohm`. `"C"` is the **coulomb**: for degrees Celsius use `degC`
  (warning if `C` appears). Hand-written parser: no `eval`.
- `expected` (optional): expected unit of the expression; the result says `expected_ok`.

Rules: sums, `Min`, `Max` and comparisons need terms of the same dimension; the arguments
of `sin`, `exp`, `log` and the other functions must be pure numbers (radians included);
a non-numeric exponent is allowed only on a dimensionless base. Equal dimensions with
different units (km/h against m/s, degrees inside `sin`) give a **warning**, not an error.

Result: `consistent`, `dimension` (exponents of the SI base quantities), `si_unit`
(`kg*m/s^2`), `scale_to_si`, `problems`; for an equation also `lhs_dimension`,
`rhs_dimension`, `lhs_si_unit`, `rhs_si_unit`, `scale_ratio` (the value in SI of the right-hand side
relative to the left; `null` if the dimensions differ).

Real examples:

```json
{"operation": "check_units", "expr": "F = 0.5*rho*v**2*Cd*A", "units": {"F": "N", "rho": "kg/m^3", "v": "m/s", "Cd": "1", "A": "m^2"}}
{"consistent": true, "dimension": {"mass": 1, "length": 1, "time": -2}, "si_unit": "kg*m/s^2", "scale_to_si": 1.0, "problems": [], "lhs_dimension": {"mass": 1, "length": 1, "time": -2}, "rhs_dimension": {"mass": 1, "length": 1, "time": -2}, "lhs_si_unit": "kg*m/s^2", "rhs_si_unit": "kg*m/s^2", "scale_ratio": 1.0}
```

```json
{"operation": "check_units", "expr": "F = rho*v*A", "units": {"F": "N", "rho": "kg/m^3", "v": "m/s", "A": "m^2"}}
{"consistent": false, "dimension": null, "si_unit": null, "scale_to_si": null, "problems": ["the two sides have different dimensions: kg*m/s^2 and kg/s"], "lhs_dimension": {"mass": 1, "length": 1, "time": -2}, "rhs_dimension": {"mass": 1, "time": -1}, "lhs_si_unit": "kg*m/s^2", "rhs_si_unit": "kg/s", "scale_ratio": null}
```

```json
{"operation": "check_units", "expr": "v = d/t", "units": {"v": "km/h", "d": "m", "t": "s"}}
{"consistent": true, "dimension": {"length": 1, "time": -1}, "si_unit": "m/s", "scale_to_si": 0.277777777777778, "problems": [], "lhs_dimension": {"length": 1, "time": -1}, "rhs_dimension": {"length": 1, "time": -1}, "lhs_si_unit": "m/s", "rhs_si_unit": "m/s", "scale_ratio": 3.6}
```

Warning of the last example: Same dimensions but different units: in SI the right-hand side is 3.6 times the left one; convert the units before using the formula.

---

## 8. `python_sandbox` (D13)

Our own interpreter of a subset of Python. **No `exec`**: the code is read with
`ast.parse` and every node is executed by the interpreter.

| Allowed | Forbidden |
|---|---|
| assignments, `if`, `for`, `while`, `break`, `continue`, `def` (with decorators `lru_cache`/`cache`), `lambda`, `return`, comprehensions, f-strings, `try`/`except`, `assert`, `del`, `global` inside a function (D38) | `class`, `with`, `yield`, `async`, `nonlocal`, `match`, attribute assignment; `global` of names with `__` or of forbidden names |
| `import`, `from ... import` and `from ... import *` (D38: only the public names that pass the same checks) of the modules: `math`, `cmath`, `fractions`, `decimal`, `itertools`, `functools`, `collections`, `statistics`, `random` (fixed seed 0), `heapq`, `bisect`, `sympy`, `numpy`, `scipy` (calculation submodules), `mpmath`; `time` only for the clocks (`time`, `perf_counter`, `monotonic`, `process_time`, `localtime`, `gmtime`, `strftime` and similar, D40); `datetime` (`date`, `datetime`, `time`, `timedelta`, `timezone`, `tzinfo`, `MINYEAR`, `MAXYEAR`, `UTC`, D40); `vars` as a variable name or argument (D40) | any other module; `time.sleep`; `open`, `eval`, `exec`, `compile`, `getattr`, `setattr`, `type`, `vars()` as a builtin, `globals`, `input` |
| `sympy.S` with constants (`S.Reals`, `S.Half`) and numbers (`S(1)`); `nsimplify` of numbers (D38: the form found by `mpmath.identify` is read by the AST parser) | `S('text')`, `nsimplify('text')`, changes to `S` |
| attributes without a leading `_` | attributes that start with `_`; names with `__`; blacklisted attributes (`format`, `format_map`, `ctypes`, `tofile`, `load`, `save`, `lambdify`, `sympify`, `parse_expr`, `preview`, `plot` and others, list in the code) |

`result`: `stdout` (printed text), `value` (representation of the last expression, or `null`),
`truncated` (boolean). Program errors: `ok: false`, `code = MATH_ERROR` with the exception type
and line.

---

## 9. `lean_check` (D16, D17)

Checks Lean 4 code with Mathlib. If the code contains no `import`, the server adds
`import Mathlib`. Lean and Mathlib live inside the repo in folders excluded from git; the path of the
Lean project can be changed with the variable `OCE_MATH_LEAN_PROJECT`.

`result`: `valid`, `errors`, `warnings`, `has_sorry`, `axioms`, `messages`, `elapsed_s`. Lean
missing: `ok: false`, `code = UNAVAILABLE`.

`valid` is true only if: Lean finishes without errors; the axiom check ran; the axioms
used by the declarations of the code are only `propext`, `Classical.choice`, `Quot.sound`
(plus `sorryAx` if `allow_sorry` is `true`). `axioms` is the sorted list of the axioms used, `null`
if the check did not run (for example a comment left open at the end).

**Security (D36).** Lean runs code while it compiles. Before calling Lean the server rejects with
`FORBIDDEN`:

- commands that run code: `#eval`, `run_cmd`, `run_tac`, `elab`, `macro`, `syntax`,
  `initialize`, `simproc` and the commands `register_*`, `declare_*`; among the commands with `#` only
  `#check`, `#check_failure`, `#print`, `#reduce`, `#synth`, `#simp`, `#norm_num`, `#guard_msgs` are allowed;
- new axioms and proofs entrusted to the compiler: `axiom`, `unsafe`, `implemented_by`, `extern`,
  `native_decide`, `decide +native`, `bv_decide`, `Lean.ofReduceBool`, hand-written `sorryAx`;
- attributes outside an allowed list (`simp`, `ext`, `instance`, `reducible`, `norm_cast`,
  `gcongr` and similar), `set_option` other than `maxHeartbeats`, `maxRecDepth`, `synthInstance.*`,
  `pp.*`, `linter.*`, `trace.*`, `autoImplicit`; `import` of modules outside `Mathlib`;
- `polyrith` (calls a network service).

The filter works on the whole text, comments included. At timeout the server kills the whole tree of
Lean processes.

---

## 10. `date_calc`

Default time zone `Europe/Rome`. Input dates are ISO (`2026-11-23`, `2026-11-23T10:30`) or
`dd/mm/yyyy`. Missing `ref_date` = today in the chosen time zone.

**National holidays** (D19): 1 January (New Year's Day), 6 January (Epiphany), Easter Sunday, Easter Monday,
25 April (Liberation Day), 1 May (Labour Day), 2 June (Republic Day), 15 August (Assumption Day), **4 October (St. Francis of Assisi) from 2026**, 1 November (All Saints' Day), 8 December (Immaculate Conception),
25 December (Christmas Day), 26 December (St. Stephen's Day). Easter with the Gauss algorithm in the anonymous Gregorian form
(Meeus/Jones/Butcher). Local patron saints are not included. **Business day** = Monday-Friday not a
holiday.

| Operation | Fields | Keys of `result` and rules |
|---|---|---|
| `now` | `timezone` | `datetime`, `date`, `time`, `weekday`, `utc_offset`, `is_dst` |
| `age` | `birth_date`, `ref_date` | `years`, `months`, `days`, `total_days`, `birthday_passed_this_year`, `next_birthday`, `days_to_next_birthday`. People born on 29 February: in non-leap years the birthday is 28 February, with a warning |
| `weekday` | `date` | `weekday` (English day name), `iso_week`, `day_of_year`, `is_holiday`, `holiday_name`, `is_business_day` |
| `diff` | `date`, `end_date` | `days` (`end - date`), `weeks`, `business_days` = business days `d` with `date < d <= end_date` |
| `add` | `date` + `years`, `months`, `weeks`, `days`, `hours`, `minutes`, `business_days` | `date`. End of month: 31 January + 1 month = 28 February (29 February in leap years). With `business_days: true` the `days` are business days |
| `resolve_relative` | `ref_date`, `target_weekday`, `which`, `nth`; or `days` with `business_days` | `date`. `next` = first after `ref_date` (excluded); `previous` = last before; `this_week` = that day in the ISO week of `ref_date`; `next_week` = in the following ISO week; `nth_next` = the `nth` after |
| `event_schedule` | `date` (local date and time), `duration_minutes`, `reminders_before` (`"48h"`, `"2d"`, `"3bd"`, `"30m"`), `timezone` | `start`, `end`, `reminders` in ISO with offset (`+01:00` CET, `+02:00` CEST). `h` and `m` = absolute time; `d` = same local time N days before; `bd` = N business days before. Non-existent time (March) moved forward with a warning; repeated time (October) = first occurrence with a warning |
| `invoice_due` | `date` (invoice date), `dffm_days`, `business_days` | `due_date`. DFFM rule: invoice date + `dffm_days` days, then the last day of that month. With `business_days: true` a due date on a holiday moves to the first business day after |
| `month_calendar` | `date` (`2026-12` or a date in the month) | `weeks` (Monday-Sunday rows), `holidays`, `business_days_count` |
| `holidays` | `date` (`2027` or a date in the year) | `holidays` (date and name), `easter`, `easter_monday` |
| `recurring` | `date` (first occurrence, with time), `days` or `weeks` or `months` or `years` (interval, only one different from 0), or `target_weekday` + `month_week` (monthly by weekday: `1` = first, `-1` = last); `count` or `end_date`; `timezone` | `occurrences` in ISO with offset. Monthly on day 29-31: in shorter months the last day of the month, with a warning. The local time stays the same even after the clock change |
| `convert_tz` | `date` (date and time in `timezone`), `to_timezone` | `source`, `target` in ISO with offset, `offset_difference_hours`. IANA time zones (`Europe/London`, `America/New_York`) |

Rules added (D24):

- `age` also gives `total_months` (completed months) and `total_weeks` (completed weeks), useful for
  small children.
- `diff` with dates and times also gives **real** `elapsed_hours` and `elapsed_minutes`: the night of the March
  clock change lasts 23 hours, the October one 25.
- `add` adds `days`, `weeks`, `months`, `years` on the **calendar** (same local time) and `hours`,
  `minutes` on **real time** (with the clock change the local time shifts), with a warning.

---

## 11. `geom3d_calc`

Angles in degrees, lengths in the user's unit (mm for CAD). Quaternions `[w, x, y, z]`.
Cylindrical coordinates `[r, theta_deg, z]`; spherical ISO 80000-2 `[r, theta_deg polar, phi_deg azimuth]`.

| Operation | Fields | Keys of `result` |
|---|---|---|
| `vec_ops` | `v1`, `v2` optional | `norm_v1`, `unit_v1`, and with `v2`: `dot`, `cross`, `norm_v2`, `sum`, `difference`, `projection_v1_on_v2` |
| `angle_between` | `v1`, `v2` | `angle_deg`, `angle_rad` |
| `rotate_point` | `v1` point; `axis` + `angles_deg[0]` (Rodrigues) or `angles_deg` `[a, b, c]` with `params.sequence` (default `"XYZ"`, fixed axes in the written order); `params.center` | `point`, `rotation_matrix` |
| `transform_4x4` | `angles_deg`, `translation`, or `params.matrices` to compose; `params.invert`; `v1` point to transform | `matrix`, `inverse`, `point` |
| `quat_euler` | `quat` or `angles_deg` (sequence `params.sequence`) | `quat`, `angles_deg`, `rotation_matrix` |
| `dh_kinematics` | `dh_table` rows `[a, alpha_deg, d, theta_deg]` (classic DH: Rz(theta) Tz(d) Tx(a) Rx(alpha)) | `matrix`, `position`, `frames` |
| `point_plane_distance` | `v1`, `plane_point`, `plane_normal` | `distance`, `signed_distance`, `foot` |
| `line_plane_intersect` | `v1` point of the line, `v2` direction, `plane_point`, `plane_normal` | `intersection` or `parallel`, `contained` |
| `point_line_distance` | `v1` point, `params.line_point`, `params.line_dir` | `distance`, `foot` |
| `coord_convert` | `v1`, `coord_from`, `coord_to` | `coords` |
| `pcd_holes` | `params.n`, `params.pcd_diameter`, `params.start_angle_deg` (0), `params.center` (`[0, 0]`), `params.clockwise` (`false`) | `holes` (list `[x, y]` to 6 decimals), `angle_step_deg` |
| `solid_properties` | `params.shape` (`cylinder`, `tube`, `box`, `sphere`, `cone`, `flange`), dimensions, `params.density` (kg/m³, optional) | `volume`, `surface_area`, `centroid`, `mass`, `inertia` (tensor at the centre of mass) |
| `section_properties` | `params.shape` (`rectangle`, `circle`, `hollow_circle`, `rect_tube`, `i_beam`, `t_section`, `c_section`), dimensions | `area`, `centroid`, `Ix`, `Iy`, `Wx`, `Wy`, `Ip`, `rx`, `ry` |

**Operations for 3D drawing (D22).** They are used to calculate before drawing with SolidCraft,
SolidWorks or CadQuery: sketch points, coordinates to pick faces and edges, expected volume and
bounding box after each step, to compare with the CAD measurement.

| Operation | `params` | Keys of `result` |
|---|---|---|
| `polygon_2d` | `points` `[[x, y], ...]` | `area`, `signed_area`, `perimeter`, `centroid`, `is_convex`, `orientation`, `Ix`, `Iy`, `Ixy` (at the centre of mass) |
| `circle_3pts` | `points` (3 points, 2D or 3D) | `center`, `radius`, `normal` (3D) |
| `arc_fillet` | `line1` `[[x, y], [x, y]]`, `line2`, `radius` | `center`, `tangent1`, `tangent2`, `start_angle_deg`, `end_angle_deg`, `arc_length` |
| `triangle_solve` | three values among `a`, `b`, `c` (sides) and `A`, `B`, `C` (angles in degrees) | sides, angles, `area`, `perimeter`, `circumradius`, `inradius`; ambiguous case: two solutions |
| `curve_points` | `curve` (`helix`, `involute`, `spline`, `parametric`, `arc`, `circle`, `ellipse`), curve parameters, `n` points | `points`, `length` |
| `intersect` | `a`, `b`: objects `{"type": "line"/"plane"/"sphere"/"cylinder", ...}` | `kind` (`point`, `points`, `line`, `circle`, `none`, `coincident`), geometry of the intersection, `closest_points` and `distance` for skew lines |
| `feature_points` | `shape` (`box`, `cylinder`), dimensions, `origin`, `rotation_deg` | `vertices`, `edge_midpoints`, `face_centers` with face name |
| `pattern` | `kind` (`linear`, `grid`, `circular`), `count`, `spacing` or `angle_deg`, `axis`, `origin` | `points` |
| `point_cloud` | 3D `points` | `bbox_min`, `bbox_max`, `size`, `center`, `centroid`, `hull_volume`, `hull_area` |
| `mesh_properties` | `vertices`, `faces` (triangles, indices from 0) | `volume`, `area`, `centroid`, `bbox_min`, `bbox_max`, `watertight`, `inertia` |
| `composite_properties` | `parts`: list of `{"shape", dimensions, "position", "op": "add"/"subtract"}`, `density` optional | `volume`, `centroid`, `bbox_min`, `bbox_max`, `mass`; the `add` parts must not overlap and the `subtract` ones must lie inside an `add` (warning if it cannot be checked) |

The exact parameters of every curve and shape are in the tests and in the docstring of the code; from server
version 1.0.0 they will also be in this spec.

`solid_properties` has no materials database: the density is passed by hand. The database
arrives with `eng_calc.material_lookup` in phase 2, with values taken from sources in the project knowledge base.

---

## 12. `eng_calc`

Schema fixed (JSON block of section 14). The 25 operations and the formulas are those of spec
V2, section 2, tool 5. D39: all are done, one area at a time; every operation documents here the
keys of `params` before the code. **Ready since 0.2.0: the 6 operations of area A** (table
below), the 6 of area B (mechanics, vibrations, transmissions, CNC), the 4 of area C (fluids and
thermal) and the 5 of area D (circuits, filters, Clarke-Park, controls, optics and acoustics). The
other 4 (`fatigue_life`, `bolt_joint`, `weld_fillet`, `material_lookup`) need data tables
with sources: ready with D44 (below). **Areas E and F (D43, spec
3.11.0)**: 8 new operations from the physics catalogue (in the project knowledge base), in the extension of the
contract approved by the maintainer on 2026-11-24: `hydrostatics`, `pump_system`, `open_channel`,
`flow_meter`, `compressible_flow` (fluids) and `ideal_gas`, `thermo_cycle`, `calorimetry`
(thermodynamics). **Operations with tables (D44, spec 3.12.0)**: `fatigue_life`,
`bolt_joint`, `weld_fillet`, `material_lookup`, with values taken from cited sources.
**Strength of materials without tables (D46, spec 3.14.0)**: `thermal_stress`, `contact_hertz`,
`spring_helical`, `interference_fit`, `strain_analysis`, from the physics catalogue.

**The formula lives in the server**: the model chooses the operation and passes the data; the server applies the
formula, puts the results in `result` in SI units, the unit of each key in `units` and the formulas
used in `formula_used`. `formatted` reports the values in technical units (MPa, mm).

Values in `params`:

- a number is in the SI unit of the quantity: m, m2, m3, m4, N, N/m, Pa, Nm, W, rad/s, K, V, A,
  ohm, H, F, Hz;
- alternatively `[value, "unit"]` with the units of `unit_convert` (section 3) plus `m4`, `cm4`,
  `mm4`, `in4` (area moment of inertia), `N/m`, `kN/m`, `N/mm` (distributed load), `rad/s`,
  `rpm`, `m/min`, `mm/min`, `mm/s` (speed), `kN/mm` (stiffness), `N*s/m`, `kg/s` (damping),
  `Hz`, `kHz`, `m/s2`, `kg*m2`, `kg*cm2`, `kg*mm2` (mass moment of inertia), `kg/m`, `g/m`; areas C and D: `g/cm3`, `kg/L`, `kg/dm3`
  (density), `Pa*s`, `mPa*s`, `cP`, `P` (dynamic viscosity), `m2/s`, `mm2/s`, `cSt`, `St`
  (kinematic viscosity), `W/(m*K)`, `W/(m2*K)`, `K/W`, `C/W`, `mV`, `kV`, `mA`, `uA`, `kA`,
  `mohm`, `kohm`, `Mohm`, `ohm*m`, `ohm*mm2/m`, `mH`, `uH`, `nH`, `mF`, `uF`, `nF`, `pF`; areas E and F: `J/(kg*K)`, `kJ/(kg*K)` (specific heat, gas constant), `J/kg`,
  `kJ/kg`, `MJ/kg` (heat per kg), `kg/mol`, `g/mol`, `mol`, `kmol`, energies (`J`, `kJ`,
  `kWh`), flow rates (`m3/h`, `L/s`, `L/min`) and volumes (`L`, `dm3`, `cm3`); D45: `ms`, `us` (times);
  temperatures as `[value, "C"]`, `"K"`, `"F"`.
  Example: `"E": [210, "GPa"]`, `"I": [800, "cm4"]`, `"n": [1450, "rpm"]`;
- a unit of the wrong type (`"E": [210, "mm"]`), an unexpected key, a non-finite number or
  a value outside the domain give `INVALID_ARGUMENT` with the reason. Pure numbers (`nu`, `K`,
  `epsilon`) do not accept units.

| Operation | Keys of `params` | Keys of `result` |
|---|---|---|
| `hooke_elastic` | `E`; optional `nu` or `G`; only one of `sigma`, `epsilon`, `F` with `A`; `L` (elongation); one of `tau` and `gamma` (needs `G` or `nu`) | `E`, `sigma`, `epsilon`, `delta_L`, `epsilon_transverse`, `nu`, `G`, `tau`, `gamma` (those that can be calculated) |
| `beam_calc` | `support` (`simply_supported`, `cantilever`, `fixed_fixed`), `load` (`point`, `uniform`), `L`, `E`, `I`; `F` and `a` for `point` (`a` from the left end, default `L/2`, for the cantilever `L`), `q` for `uniform`; optional `W` or `c` for `sigma_max` | `R_A`, `R_B`, `M_A`, `M_B`, `M_mid`, `V_max`, `M_max`, `x_M_max`, `f_max`, `x_f_max`, `sigma_max` |
| `shaft_torsion_bending` | `T` or `P` with `n`; `M` (bending, default 0); `d` (and `d_i` for the hollow shaft) or `sigma_allow` for `d_min` of the solid shaft; `criterion` (`von_mises` default, `tresca`); `L` with `G` for the rotation | `T` (if from `P`), `d_min`, `J`, `I`, `tau_max`, `sigma_b`, `sigma_eq`, `theta`, `theta_deg`, `SF` (`sigma_allow / sigma_eq`) |
| `stress_criterion` | `sx`, `sy`, `sz`, `txy`, `tyz`, `tzx` (default 0, also negative); `Sy` optional | `principal` (`[s1, s2, s3]`, descending), `tau_max`, `sigma_vm`, `sigma_tresca`, `angle_deg` (plane state), `SF_vm`, `SF_tresca` (`null` if the stress is zero) |
| `euler_buckling` | `E`, `I`, `L`; `end` (`pinned_pinned` K = 1, `fixed_free` K = 2, `fixed_pinned` K = pi / 4.4934, `fixed_fixed` K = 0.5) or `K`; optional `A`, `Sy`, `F` | `K`, `L0`, `P_cr_euler`, `P_cr`, `method` (`euler`, `johnson`), `r`, `slenderness`, `sigma_cr`, `slenderness_limit`, `SF` |
| `pressure_vessel` | `p` (internal) and `p_o` (external, default 0); `d_i` or `r_i`; one of `t`, `d_o`, `r_o`; `shape` (`cylinder` default, `sphere`); `ends` (`closed` default, `open`); `Sy` optional | `r_i`, `r_o`, `t`, `t_over_r_i`, `sigma_h_thin`, `sigma_a_thin`, `sigma_t_inner`, `sigma_t_outer`, `sigma_r_inner`, `sigma_r_outer`, `sigma_a`, `sigma_vm_inner`, `SF` |

Area B:

| Operation | Keys of `params` | Keys of `result` |
|---|---|---|
| `kinematics_dynamics` | `mode` plus the keys of the mode: `projectile` (`v0`, `angle_deg`, `h0` default 0, `g` default 9.80665); `uniform_acceleration` (3 of `v0`, `v`, `a`, `t`, `s`); `energy` (`m` with `v` and/or `h`, `k` with `x`, `I_m` with `omega`, `g`); `rotation` (2 of `P`, `T`, `omega`); `steiner` (`I_m`, `m`, `d`); `circular` (`r`, `v` or `omega`, `m` optional) | `projectile`: `t_flight`, `range`, `t_apex`, `h_max`, `v_impact`, `impact_angle_deg`; `uniform_acceleration`: the 2 unknowns (lists if there is more than one solution with t >= 0); `energy`: `E_kin`, `E_pot`, `E_spring`, `E_rot`, `E_total`; `rotation`: the missing quantity; `steiner`: `I_axis`; `circular`: `v`, `omega`, `a_c`, `period`, `F_c` |
| `vibration_1dof` | `m`, `k`; one of `c` and `zeta`; optional forcing `omega` or `f` and its amplitude `F0` | `omega_n`, `f_n`, `T_n`, `c_c`, `c`, `zeta`, `regime` (`underdamped`, `critically damped`, `overdamped`), `omega_d`, `f_d`, `T_d`, `log_decrement`, `Q` (`null` where not defined); with forcing `r`, `amplification`, `phase_deg`, `transmissibility`, with `F0` also `X`, `F_T` |
| `bearing_life` | `type` (`ball` p = 3, `roller` p = 10/3) or `p`; `P` or `Fr` with `Fa`, `X`, `Y`; `C`; optional `n`, `a1`, `L10h_req` (with `n`: gives `C_required`) | `P`, `p`, `L10` (revolutions), `L10_mrev`, `L10h` (hours), `L_na`, `L_nah`, `C_required` |
| `belt_drive` | `D` (large pulley), `d`, `C` (centre distance); optional `n` (small pulley), `P` (with `n`), `mu`, `groove_deg` (V-groove), `m_l` (mass per metre) | `L` (exact length), `theta_small`, `theta_small_deg`, `theta_large`, `ratio`, `v`, `n_large`, `F_eff`, `mu_eff`, `tension_ratio`, `T1`, `T2`, `T0`, `F_centrifugal` |
| `cnc_machining` | `mode` (`turning`, `milling`, `drilling`), `D`, one of `Vc` and `n`; turning `f`, `ap`; milling `z`, `fz`, `ap`, `ae`; drilling `f`; optional `kc` (specific cutting pressure), `eta` | `n`, `n_rpm`, `Vc`, `Vc_m_min`, `Vf`, `Vf_mm_min`, `MRR`, `MRR_cm3_min`, `F_c` (turning), `P_cut`, `T_spindle`, `P_motor` |
| `gear_spur` | `m` (normal module), `z1`, `z2` (integers), `alpha_deg` (default 20), `beta_deg` (helix, default 0); optional `n1`, `T1` or `P` with `n1`, `b` with `Y` (Lewis) | `ratio`, `m_t`, `d1`, `d2`, `da1`, `da2`, `df1`, `df2`, `db1`, `db2`, `a`, `contact_ratio`, `n2`, `n2_rpm`, `T1`, `T2`, `Ft`, `Fr`, `Fa`, `sigma_lewis`, `overlap_ratio` |

Area C:

| Operation | Keys of `params` | Keys of `result` |
|---|---|---|
| `fluid_pipe` | `D`, one of `Q` and `v`, `rho`, one of `mu` and `nu`; optional `eps` (roughness, default 0), `L` (distributed losses), `K` (sum of the local loss coefficients), `dz` (outlet height minus inlet height), `p1` (upstream pressure) | `A`, `v`, `Q`, `Re`, `regime` (`laminar`, `transitional`, `turbulent`), `relative_roughness`, `f`, `dp_friction`, `h_friction`, `dp_local`, `h_local`, `dp_total`, `H_pump`, `P_hydraulic`, `p2` (those that can be calculated) |
| `aero_hydro_force` | `rho`, `A`; `v` (forces) and/or `m` (terminal fall velocity, needs `Cd`); optional `Cd`, `Cl`, one of `T` (air) and `c` (speed of sound) | `q`, `F_D`, `P_drag`, `F_L`, `c_sound`, `Mach`, `v_terminal` |
| `heat_transfer` | `mode` plus the keys of the mode: `wall` (`layers` = list of `[thickness m, conductivity W/(m K)]`, `A`, `T1`, `T2`, optional `h_in`, `h_out`); `cylinder` (`r1`, `r2`, `k`, `L`, `T1`, `T2`, optional `h_in`, `h_out`); `convection` (`h`, `A`, `T1`, `T2`); `radiation` (`emissivity`, `A`, `T1`, `T2`); `lmtd` (`flow` = `counter` or `parallel`, `Th_in`, `Th_out`, `Tc_in`, `Tc_out`, `U`, one of `Q` and `A`) | `wall`: `R_total`, `U`, `Q`, `q`, `T_interfaces`; `cylinder`: `R_total`, `Q`, `Q_per_m`; `convection`: `R_total`, `Q`, `q`; `radiation`: `Q`, `q`; `lmtd`: `LMTD`, `Q`, `A_required` (with `Q`) or `A`, `C_hot`, `C_cold` |
| `heatsink_thermal` | `P`, `T_a`, `R_jc`; optional `R_cs` (default 0), `R_sa`, `T_j_max` (at least one of `R_sa` and `T_j_max` is needed) | `R_total`, `T_j`, `T_j_C`, `T_case`, `T_case_C`, `T_sink`, `T_sink_C`; with `R_sa` and `T_j_max` also `margin_K`; without `R_sa` also `R_sa_required` |

Area D:

| Operation | Keys of `params` | Keys of `result` |
|---|---|---|
| `circuit_ac_dc` | `mode` plus the keys of the mode: `ohm` (2 of `V`, `I`, `R`, `P`); `resistors` (`values` = list in ohm, `connection` = `series` or `parallel`); `divider` (`Vin`, `R1`, `R2`); `ac_single` and `ac_three` (`V`, one of `I` and `P`, one of `cos_phi` and `phi_deg`); `cable_drop` (`phases` = `single` or `three`, `I`, `L`, `A_section`, one of `material` = `copper` and `rho`, optional `cos_phi` default 1, `V`) | `ohm`: `V`, `I`, `R`, `P`; `resistors`: `R_eq`; `divider`: `Vout`, `I`; `ac_*`: `V`, `I`, `S`, `P`, `Q`, `cos_phi`, `phi_deg`; `cable_drop`: `rho`, `R_conductor`, `dV`, `P_loss`, `dV_pct` (with `V`) |
| `rlc_filter` | `type` (`rc_lowpass`, `rc_highpass`, `rl_lowpass`, `rl_highpass`, `rlc_bandpass`), `R` with `C` and/or `L` depending on the type; optional `f` | `fc` (RC and RL) or `f0`, `Q`, `bandwidth`, `zeta` (RLC); with `f` also `gain`, `gain_dB`, `phase_deg` |
| `foc_clarke_park` | `mode` (`forward`: `ia`, `ib`, `ic` optional = `-ia - ib`; `inverse`: `d`, `q`, `zero` default 0), `theta_deg`; optional `scaling` (`amplitude` default, `power`) | `forward`: `alpha`, `beta`, `zero`, `d`, `q`; `inverse`: `alpha`, `beta`, `a`, `b`, `c` |
| `control_system` | `mode` plus the keys of the mode: `second_order` (`zeta` > 0, `wn`); `margins` (`num`, `den` = coefficients of the polynomials in s from the highest degree, open loop); `pid_zn` (`method` = `ultimate` with `Ku`, `Tu` or `step` with `K`, `L`, `T`; `controller` = `P`, `PI`, `PID` default) | `second_order`: `wd`, `Mp_pct`, `tp`, `ts_2pct`, `ts_2pct_approx`, `tr_10_90`; `margins`: `wc_gain`, `phase_margin_deg`, `wc_phase`, `gain_margin_dB` (`null` if not defined); `pid_zn`: `Kp`, `Ti`, `Td`, `Ki`, `Kd` |
| `optics_acoustics` | `mode` plus the keys of the mode: `snell` (`n1`, `n2`, `theta1_deg`); `thin_lens` (2 of `f`, `d_o`, `d_i`); `db` (`kind` = `power` or `amplitude`, one of `ratio` and `dB`); `spl` (one of `p` and `SPL_dB`); `sum_levels` (`levels` = list in dB); `inverse_square` (`L1`, `r1`, `r2`); `wave` (`f`, `v`) | `snell`: `theta2_deg` (`null` in total reflection), `total_internal_reflection`, `critical_angle_deg`; `thin_lens`: `f`, `d_o`, `d_i`, `magnification`; `db`: `dB` or `ratio`; `spl`: `SPL_dB` or `p`; `sum_levels`: `L_total_dB`; `inverse_square`: `L2_dB`; `wave`: `wavelength`, `period` |

Area E (fluids, D43):

| Operation | Keys of `params` | Keys of `result` |
|---|---|---|
| `hydrostatics` | `mode` plus the keys of the mode: `pressure` (`rho`, `h` depth; optional `p0` pressure at the free surface, default 0, and `g`); `buoyancy` (`rho` of the fluid, `V` volume of the body; optional `m` mass of the body and `g`); `plane_wall` (`rho`; `shape` = `rectangle` with `b` width and `L` length along the wall, or `circle` with `D`; `theta_deg` angle between wall and free surface, 90 = vertical wall; optional `y_top` distance of the top edge from the free surface measured along the wall, default 0, and `g`) | `pressure`: `p_hydro`, `p`; `buoyancy`: `F_b` (body fully submerged), with `m` also `W`, `F_net`, `rho_body`, `floats`, `submerged_fraction`; `plane_wall`: `A`, `I_xc`, `y_c`, `h_c`, `F`, `y_cp`, `h_cp`, `cp_offset` |
| `pump_system` | `mode` plus the keys of the mode: `power` (`Q`, `rho`, one of `H` and `dp`; optional `eta`, default 1, `n` and `g`); `affinity` (`n1`, `n2`, at least one of `Q1`, `H1`, `P1`; optional `D1` and `D2`, default equal, `rho1` and `rho2`, default equal); `npsh` (`p_s` absolute pressure at the free surface, `p_v` vapour pressure of the liquid, `rho`, `z_s` height of the free surface above the pump axis, negative if the pump draws from below; optional `h_loss` losses of the suction pipe, `NPSH_r` and `g`); `cylinder` (`D` bore, `p`; optional `d` rod, `Q` flow rate, `eta` efficiency, default 1) | `power`: `H`, `dp`, `P_hydraulic`, `P_shaft`, `T_shaft` (with `n`); `affinity`: `Q2`, `H2`, `P2` (those given at point 1), `speed_ratio`, `diameter_ratio`; `npsh`: `NPSH_a`, `margin` (with `NPSH_r`); `cylinder`: `A_bore`, `F_push`, `v_out` (with `Q`), with `d` also `A_annulus`, `F_pull`, `v_in`, `area_ratio` |
| `open_channel` | `shape` (`rectangular` with `b`; `trapezoidal` with `b` bottom and `z` side slope, horizontal over vertical; `circular` with `D`, partially full pipe), `n` Manning coefficient, `S` bed slope; one of `y` depth and `Q` (gives the normal depth); optional `g` | `y`, `A`, `P_wet`, `R_h`, `T_top`, `v`, `Q`, `Fr`, `regime` (`subcritical`, `critical`, `supercritical`), `y_critical`; `circular` also `Q_max` and `y_Q_max` |
| `flow_meter` | `type` plus the keys of the type: `orifice`, `nozzle`, `venturi` (`D` pipe, `d` orifice or throat, `Cd`, `rho`, one of `dp` and `Q`; optional `eps` expansibility, default 1 for liquids); `weir_rect` (`b`, `Cd`, one of `h` head and `Q`; optional `g`); `weir_v` (`theta_deg` opening, `Cd`, one of `h` and `Q`; optional `g`) | pipes: `beta`, `A_throat`, `Q`, `dp`, `v_throat`, `v_pipe`, `m_dot`; weirs: `Q`, `h` |
| `compressible_flow` | `mode` plus the keys of the mode, all with `gamma` (default 1.4): `isentropic` (`M`; optional `T0`, `p0`, `R` default air); `area_mach` (`A_ratio` = A/A* >= 1, `branch` = `subsonic` or `supersonic`; optional `T0`, `p0`, `R`); `normal_shock` (`M1` > 1); `choked` (`A` throat area, `p0`, `T0`; optional `R`) | `isentropic` and `area_mach`: `M`, `T_T0`, `p_p0`, `rho_rho0`, `A_Astar`; with `T0` also `T`, `a`, `V`; with `p0` also `p`; with `p0` and `T0` also `rho`, `mass_flux`; `normal_shock`: `M2`, `p2_p1`, `T2_T1`, `rho2_rho1`, `p02_p01`; `choked`: `m_dot`, `p_star`, `T_star`, `V_star` |

Area F (thermodynamics, D43):

| Operation | Keys of `params` | Keys of `result` |
|---|---|---|
| `ideal_gas` | `mode` plus the keys of the mode: `state` (three of `p`, `V`, `T` and the amount of gas, `m` or `n_mol`; optional `R` gas constant, default air, or `M_molar`); `process` (`process` = `isothermal`, `isobaric`, `isochoric`, `isentropic`, `polytropic` with `n_poly`; `p1`, `T1`, one of `V1` and `m`; one of `p2`, `V2`, `T2`; optional `gamma`, default 1.4, and `R`) | `state`: `p`, `V`, `T`, `m`, `n_mol`, `rho`, `R`, `M_molar`; `process`: `m`, `p1`, `V1`, `T1`, `p2`, `V2`, `T2`, `W` (work done by the gas), `Q` (heat received by the gas), `dU`, `dH`, `dS`, `n_poly` |
| `thermo_cycle` | `cycle` plus the keys of the cycle, all with `gamma` (default 1.4) and `R` (default air): `carnot` (`T_hot`, `T_cold`); `otto` (`r` compression ratio; optional `T1` with one of `T3` and `q_in`, and `p1`); `diesel` (`r`, `rc` cut-off ratio; optional `T1` and `p1`); `brayton` (`rp` pressure ratio; optional `T1` with `T3`, `eta_c` and `eta_t` isentropic efficiencies, default 1, and `p1`) | `efficiency`; `carnot` also `COP_refrigerator`, `COP_heat_pump`; with the temperatures: `T` (list of states 1-4), `q_in`, `q_out`, `w_net`, `efficiency_carnot`; with `p1` also `p` (list) and `mep` (Otto, Diesel); `brayton` also `back_work_ratio` |
| `calorimetry` | `mode` plus the keys of the mode: `sensible` (`m`, `c`, one of `dT` and the pair `T1`, `T2`); `latent` (`m`, `L`); `mixture` (`bodies` = list of at least 2 objects `{"m": ..., "c": ..., "T": ...}`) | `sensible`: `Q`, `dT`; `latent`: `Q`; `mixture`: `T_eq`, `T_eq_C`, `Q` (list: heat received by each body, negative if it gives it up) |

Operations with data tables (D44):

| Operation | Keys of `params` | Keys of `result` |
|---|---|---|
| `fatigue_life` | `mode` plus the keys of the mode: `endurance` (`Sut`; `surface` = `ground`, `machined`, `hot_rolled`, `as_forged`; `load` = `bending` default, `axial`, `torsion`; for bending and torsion `d` between 2.79 and 51 mm or `kb`; optional `reliability` in % among 50, 90, 95, 99, 99.9, 99.99 (default 50), `kd` and `kf` (default 1), `Se_prime`); `mean_stress` (`sigma_a`, `sigma_m`, `Se`, `Sut`; optional `Sy` for Soderberg); `sn_life` (`sigma_a`, `Sut`, `Se`, `f` fraction of `Sut` at 1000 cycles); `ec3` (`delta_sigma` with `delta_sigma_C` for normal stresses, or `delta_tau` with `delta_tau_C` for shear; optional `gamma_Ff` and `gamma_Mf`, default 1); `miner` (`blocks` = list of `[n_i, N_i]`) | `endurance`: `Se_prime`, `ka`, `kb`, `kc`, `kd`, `ke`, `kf`, `Se`; `mean_stress`: `n_goodman`, `n_gerber`, `n_soderberg` (with `Sy`), `sigma_ar_goodman`; `sn_life`: `a`, `b`, `N` (`null` = infinite life); `ec3`: `N` (`null` = below the cut-off limit), `delta_sigma_D` and `delta_sigma_L`, or `delta_tau_L`; `miner`: `D`, `remaining`, `ok` |
| `bolt_joint` | `size` (text `"M12"`: ISO 262 coarse pitch, from M1 to M64) or `d` with `P`; `grade` as text (`"4.6"`, `"4.8"`, `"5.6"`, `"5.8"`, `"6.8"`, `"8.8"`, `"10.9"`, `"12.9"`); optional `Fv` (shear) and `Ft` (tension) per bolt, `shear_planes` (default 1), `threads_in_shear` (`true`), `countersunk` (`false`), `gamma_M2` (1.25), `K` tightening factor, `mu_thread` with `mu_head` and `D_km` (VDI 2230 torque), `mu_slip` with `ks` (1) and `gamma_M3` (1.25) | `d`, `P`, `d2`, `d3`, `As`, `A`, `fyb`, `fub`, `Fv_Rd`, `Ft_Rd`, `Fp_C`; with `K` also `T_K`; with the friction coefficients also `T_vdi`; with `mu_slip` also `Fs_Rd`; with the loads `util_shear`, `util_tension`, `util_combined`, `ok` |
| `weld_fillet` | `a` throat or `z` leg of the equal-leg weld; `steel` (`S235`, `S275`, `S355`, `S420`, `S460`) with optional `t`, or `fu` with `beta_w`; optional `gamma_M2` (1.25); for the check `F` with `L` (simplified method) or `sigma_perp`, `tau_perp`, `tau_par` (directional method) | `a`, `fu`, `beta_w`, `fvw_d`, `Fw_Rd` (resistance per metre of weld); with `L` also `F_Rd`; simplified `util`; directional `sigma_eq`, `limit_eq`, `limit_perp`, `util_eq`, `util_perp`; with a check `ok` |
| `material_lookup` | `material` (`S235`, `S275`, `S355`, `C45+N`, `C45+QT`, `AISI304`, `Al6061-T6`, `Ti6Al4V`, `Cu`, `C25/30`, `C24`, `PLA`, `ABS`); optional `t` thickness or diameter (steels) | `material`, `properties` (values in SI; a range is `[min, max]`; `null` = not documented), `source` (URL of the source of each property), `note` |

Modes added by D45 to three existing operations (JSON schemas unchanged):

| Operation and mode | Keys of `params` | Keys of `result` |
|---|---|---|
| `kinematics_dynamics`, `collision` | `m1`, `v1`; optional `m2` (without `m2` the collision is against a fixed wall or one moving at `v2`), `v2` (default 0), `e` coefficient of restitution between 0 and 1 (default 1) | `v1_after`, `v2_after`, `J` (impulse on body 1), `KE_before`, `KE_after`, `KE_lost` (the energies only with `m2`) |
| `kinematics_dynamics`, `orbit` | `mu` (G M) or `M`; then `r` (circular orbit, optional `r_target` for the Hohmann transfer) or `r_p` with `r_a` (ellipse) | circular: `v_circular`, `v_escape`, `period`, `energy`; with `r_target`: `dv1`, `dv2`, `dv_total`, `t_transfer`; ellipse: `a`, `eccentricity`, `v_p`, `v_a`, `h`, `period`, `energy` |
| `kinematics_dynamics`, `gyroscope` | `I_m` (moment of inertia about the spin axis), `omega`; one of `T` (torque), `Omega_p` (precession speed) and the pair `m` with `r` (weight with arm, optional `g`) | `L`, `T`, `Omega_p`, `precession_period` |
| `kinematics_dynamics`, `flywheel` | `omega_max` with `omega_min` and the flywheel (`I_m`, or `shape` = `disk` or `ring` with `m` and `R`); or the design with `dE`, `omega_mean`, `Cs` | `I_m`, `E_max`, `E_min`, `dE`, `omega_mean`, `Cs`; design: `I_required` |
| `optics_acoustics`, `doppler` | `f`, `c` wave speed; optional `v_source` (positive towards the observer) and `v_observer` (positive towards the source), default 0 | `f_observed`, `ratio` |
| `optics_acoustics`, `string` | `L`, `T` tension, `mu` mass per metre; optional `n` harmonic (default 1) | `v`, `f_n`, `wavelength` |
| `optics_acoustics`, `pipe` | `L`, `c`, `ends` = `open_open` or `open_closed`; optional `n` (default 1) | `f_n`, `wavelength` |
| `optics_acoustics`, `intensity` | `W` sound power, `r`; optional `Q_dir` directivity (default 1), `rho` with `c` | `I`, `L_I_dB`, `L_W_dB`; with `rho` and `c` also `Z`, `p_rms`, `SPL_dB` |
| `circuit_ac_dc`, `impedance` | `f`, `elements` = list of objects `{"type": "R"/"L"/"C", "value": ...}`, `connection` (`series`, `parallel`); optional `V` (rms value) | `Z_re`, `Z_im`, `Z_abs`, `phase_deg`, `X` (reactance of each element); with `V` also `I`, `P`, `Q`, `S`, `cos_phi` |
| `circuit_ac_dc`, `transient` | `circuit` = `RC` (`R`, `C`), `RL` (`R`, `L`), `RLC` (`R`, `L`, `C`, in series); `V` step; optional `kind` = `charge` (default) or `discharge` for RC and RL, `t` | `tau` (RC, RL) or `alpha`, `omega0`, `zeta`, `regime`, `omega_d` (RLC); with `t` the value at that instant: `v_C` (RC, RLC), `i` (RL, RLC) |

D46 operations (strength of materials without tables, spec 3.14.0):

| Operation | Keys of `params` | Keys of `result` |
|---|---|---|
| `thermal_stress` | `mode` = `free`, `bar`, `plate`; `alpha` (coefficient of linear expansion); `dT` or `T1` with `T2`. `free`: optional `L`. `bar`: `E`; optional `A`, `L`, `gap` (total gap between the bar and two rigid walls, needs `L`). `plate`: `E`, `nu` | `dT`, `epsilon_th`; `free`: `dL` (with `L`); `bar`: `sigma`, `F` (with `A`), `delta_free` (with `L`), `gap_closed` (with `gap`); `plate`: `sigma` |
| `contact_hertz` | `geometry` = `sphere` or `cylinder` (cylinders with parallel axes); `F`, `R1`; optional `R2` (positive convex, negative for a concave seat with `|R2| > R1`, absent for a plane); `E1`, `nu1`; optional `E2` with `nu2` (absent = same material); `cylinder`: `L` | `R`, `E_star`, `p0` (maximum pressure), `p_mean`, `tau_max_1`, `z_tau_max_1`, `tau_max_2`, `z_tau_max_2` (maximum shear stress below the surface and depth, with the `nu` of each body); `sphere`: `a` (contact radius), `d` (approach); `cylinder`: `b` (half-width) |
| `spring_helical` | compression spring of round wire: `d` (wire), `D` (mean diameter) or `D_o` (outer), `Na` (active coils), `G`; optional `F` or `deflection`, `ends` = `plain`, `plain_ground`, `squared`, `squared_ground`, `L0` (free length, needs `ends`), `rho` (wire density) | `D`, `C`, `k`, `K_W` (Wahl), `K_B` (Bergsträsser); with a load `F`, `deflection`, `tau` (with `K_B`), `tau_wahl`; with `ends` `Nt`, `L_solid`; with `L0` `F_solid`, `tau_solid`; with `rho` `m_active`, `f_n` |
| `interference_fit` | `d` (nominal diameter of the contact), `D_o` (outer of the hub), optional `d_i` (bore of the hollow shaft); `delta` **diametral** interference; `E_hub`, `nu_hub`; optional `E_shaft` with `nu_shaft` (absent = same material); optional `mu` with `L` (length of the contact), `alpha_hub` (shrink fit) with optional `clearance` (assembly clearance) | `p` (contact pressure), `sigma_t_hub`, `sigma_r_hub`, `sigma_vm_hub`, `sigma_t_shaft`, `sigma_r_shaft`; with `d_i` `sigma_t_shaft_inner`; with `mu` and `L` `F_axial`, `T`; with `alpha_hub` `dT_assembly` |
| `strain_analysis` | `mode` = `transform`: `ex`, `ey`, `gxy` (engineering shear strain), optional `theta_deg`; `mode` = `rosette`: `ea`, `eb`, `ec` with `rosette` = `rectangular` (0/45/90 degrees) or `delta` (0/60/120 degrees) or `angles_deg` (any 3 angles). Optional `E` with `nu` for the stresses | `e1`, `e2`, `gamma_max`, `theta_p_deg`; `rosette`: also `ex`, `ey`, `gxy`; with `theta_deg` `ex_theta`, `ey_theta`, `gxy_theta`; with `E` and `nu` `sigma1`, `sigma2`, `tau_max`, `e3` |

D47 operations (basic sciences, spec 3.15.0):

| Operation | Keys of `params` | Keys of `result` |
|---|---|---|
| `physical_constant` | `name` (CODATA name or short symbol, as `const()` of `math_eval`) or `search` (text searched in the names) | with `name`: `name`, `symbol`, `value`, `value_text` (decimal notation), `unit`, `uncertainty` (standard uncertainty), `relative_uncertainty`, `exact`, `codata` (edition), `source`; with `search`: `matches` (at most 25 names), `count` |

D48 operations (chemistry, spec 3.16.0). Concentrations in **mol/L** (`M`, `mmol/L`, `umol/L`,
`mol/m3` accepted), volumes in m3 in the result:

| Operation | Keys of `params` | Keys of `result` |
|---|---|---|
| `molar_mass` | `formula` (e.g. `Ca(OH)2`, `K4[Fe(CN)6]`, `CuSO4·5H2O`; hydrates with `·`, `*`, `.`); optional `m` or `n` | `formula`, `elements` (atoms per element), `M` (kg/mol), `M_g_mol`, `M_uncertainty_g_mol`, `mass_fraction`; with `m` `n`, with `n` `m` |
| `reaction_stoich` | `reaction` (text with `->`, `→`, `=`, `<=>` or `⇌`, species separated by `+`, written coefficients ignored); optional `amounts` = object `{"species": amount}` on the reactants: number in mol, `[value, "g"]` (mass) or `[value, "mmol"]` | `reactants`, `products`, `species`, `coefficients` (smallest integers), `balanced`, `M_g_mol`; with `amounts` `limiting`, `extent`, `n_reacted`, `mass_reacted`, `n_left` |
| `acid_base` | `mode` = `strong`: `kind` (`acid`, `base`), `c`; `weak`: `kind`, `c`, `Ka` or `pKa` (acid), `Kb` or `pKb` (base); `buffer`: `c_acid`, `c_base`, `Ka` or `pKa`. Optional `Kw` (default 1.0e-14 at 25 degrees C, with a warning) | `pH`, `pOH`, `H`, `OH`; `strong` and `weak`: `pH_approx`; `weak`: `alpha` (degree of dissociation); `buffer`: `pH_henderson` |
| `solution_prep` | `mode` = `dilution`: three of `c1`, `V1`, `c2`, `V2`; `molarity`: `V` with `c` or with the amount (`n`, or `m` with `formula` or `M`) | `dilution`: the missing quantity and `n`; `molarity`: `c`, `n`; with the molar mass also `m`, `M` |
| `reaction_kinetics` | `mode` = `arrhenius`: `A`, `Ea`, `T` or `k1`, `T1`, `k2`, `T2` or `k1`, `T1`, `Ea`, `T2`; `integrated`: `order` (0, 1, 2), `k`, `c0`, and `t` or `c` | `arrhenius`: `k`, or `Ea` with `A`, or `k2` with `A`; `integrated`: `c`, `t`, `t_half` |
| `electrochemistry` | `mode` = `nernst`: `E0`, `n`, `Q`, optional `T`; `electrolysis`: `charge` or `I` with `t`, `n` (electrons per ion), and `formula` or `M`; `gibbs`: `n` and one of `E`, `dG`, `K`, optional `T` (default 298.15 K) | `nernst`: `E`, `slope_per_decade`, `T`; `electrolysis`: `charge`, `n_mol`, `m`; `gibbs`: `E`, `dG`, `K`, `lnK`, `T` |


D49 operations (nuclear and modern physics, special relativity, spec 3.17.0). Nuclides written
as `"U-235"`, `"235U"`, `"Tc-99m"`, `"alpha"`, `"p"`, `"n"`, `"d"`, `"t"`:

| Operation | Keys of `params` | Keys of `result` |
|---|---|---|
| `nuclide_data` | `nuclide` | `nuclide`, `Z`, `N`, `A`, `symbol`, `isomer`, `mass_excess_keV`, `mass_excess_unc_keV`, `estimated`, `atomic_mass_u`, `atomic_mass_text`, `atomic_mass_unc_u`, `atomic_mass_kg`, `nuclear_mass_MeV`, `binding_energy_MeV`, `binding_per_nucleon_MeV`, `excitation_keV`, `stable`, `half_life_s`, `half_life_text`, `spin_parity`, `decay_modes`, `abundance_percent`, `source`; if unstable `decay_constant`, `mean_life_s` |
| `radioactive_decay` | `mode` = `law`: one of `nuclide`, `T_half`, `lambda`, `tau`; `t` or `fraction` (N/N0); optional one of `N0`, `A0`, `m0` (with `m0` the nuclide or `M`). `activity`: as above for the time; one of `m`, `N`, `A`; optional `M`. `chain`: `parent` or `T_half1`, `daughter` or `T_half2`, `N1_0` or `A1_0`, `t`; optional `N2_0`, `branching` | `law`: `lambda`, `T_half`, `tau`, `t`, `fraction`, `decayed_fraction`, `half_lives`; with the amount `N0`, `N`, `A0`, `A` (and `m0`, `m`). `activity`: `N`, `A`, `A_Ci`; with the molar mass `m`, `specific_activity`, `M`. `chain`: `lambda1`, `lambda2`, `N1`, `N2`, `A1`, `A2`, `activity_ratio`, `equilibrium` (`secular`, `transient`, `none`), `equilibrium_ratio`, `t_max_daughter` |
| `nuclear_reaction` | `reaction` (text with `->`; coefficients like `3 n`; leptons `e-`, `e+`, `nu`, `anti-nu`, `gamma`); optional `projectile` (for the threshold) | `reaction`, `Q_MeV`, `Q_keV`, `Q_J`, `Q_unc_keV`, `mass_defect_u`, `exothermic`; with Q > 0 `energy_per_kg_reactants_J`; with Q < 0 and two reactants `T_threshold_MeV`, `T_threshold_classical_MeV`; two-body decay `T_products_MeV` |
| `modern_physics` | `mode` = `photon`: one of `wavelength`, `frequency`, `energy`. `photoelectric`: `work_function` or `stopping_voltage`, and the light (`wavelength`, `frequency`, `photon_energy`). `compton`: `wavelength` or `energy`, `angle_deg`, optional `particle`. `de_broglie`: `particle` or `m` (with `charge`), one of `v`, `kinetic_energy`, `p`, `voltage`. `hydrogen_like`: optional `Z`, `nuclide` (reduced mass); `n`, or `n1` and `n2`, or `n1` and `wavelength`. `blackbody`: `T`, optional `wavelength`, `wavelength1` with `wavelength2`, `emissivity`, `area`. `uncertainty`: one of `dx`, `dp`, `dE`, `dt`, optional mass | `photon`: `wavelength`, `frequency`, `energy_J`, `energy_eV`, `momentum`, `wavenumber`. `photoelectric`: `work_function_eV`, `f_cutoff`, `wavelength_cutoff`; with the light `photon_energy_eV`, `emits`, `K_max_eV`, `stopping_voltage`. `compton`: `wavelength`, `wavelength_scattered`, `shift`, `max_shift`, `compton_wavelength`, `energy_eV`, `energy_scattered_eV`, `electron_kinetic_eV`, `recoil_angle_deg`. `de_broglie`: `wavelength`, `wavelength_classical`, `p`, `p_MeV_c`, `v`, `beta`, `gamma`, `kinetic_energy_eV`. `hydrogen_like`: `Z`, `reduced_mass_factor` and `E_n_eV`, `ionization_energy_eV`, `r_n`, `v_n`, or `n_lower`, `n_upper`, `E_lower_eV`, `E_upper_eV`, `delta_E_eV`, `wavelength`, `frequency`, `series`, or `photon_energy_eV`, `ionizes`, `n2_exact`, `n2`, `E_n2_eV` (or `ejected_kinetic_eV`). `blackbody`: `wavelength_max`, `frequency_max`, `exitance`; `power`, `spectral_radiance`, `spectral_exitance`, `band_fraction`, `band_exitance`. `uncertainty`: `dp_min` or `dx_min` (with the mass `dv_min`), `dt_min`, `dE_min_J`, `dE_min_eV` |
| `special_relativity` | `mode` = `kinematics`: one of `v`, `beta`, `gamma`, `kinetic_energy`, `p`, `total_energy`; `particle` or `m` (needed with the energies); optional `proper_time`, `proper_length`. `velocity_addition`: `u` or `u_beta`, `v` or `v_beta` (signed). `doppler`: `f` or `wavelength` with `beta` or `v` (positive if the source recedes), or `z`, or source and observed. `lorentz`: `x`, `t`, `v` or `beta`. `mass_energy`: `m` or `particle`, or `energy` | `kinematics`: `beta`, `v`, `gamma`; with the mass `E0_J`, `E0_MeV`, `E_J`, `E_MeV`, `K_J`, `K_MeV`, `K_classical_J`, `p`, `p_MeV_c`; `time`, `length`. `velocity_addition`: `beta`, `v`, `beta_classical`. `doppler`: `beta`, `z`, `receding`, `f_observed` or `wavelength_observed` (or `v`). `lorentz`: `gamma`, `x_prime`, `t_prime`, `interval_s2`, `interval_type`. `mass_energy`: `m`, `m_u`, `energy_J`, `energy_MeV`, `energy_kWh` |

Particle names accepted by `particle`, `nuclide` and `reaction`: only the English ones (`electron`, `positron`, `proton`, `neutron`, `deuteron`, `triton`, `muon`); the Italian aliases were removed (D61).

D50 operations (water and steam with IAPWS-IF97, spec 3.18.0). Pressures in Pa or
`[value, "MPa"]`, `"bar"`, `"kPa"`; temperatures in K or `[value, "C"]`; `h` in J/kg or
`"kJ/kg"`, `s` in J/(kg K) or `"kJ/(kg*K)"`, `v` in m3/kg or `"L/kg"`, `rho` in kg/m3. The `phase` text is one of: `compressed liquid`, `superheated vapour`, `high-temperature vapour`, `liquid-vapour mixture`, `supercritical fluid`:

| Operation | Keys of `params` | Keys of `result` |
|---|---|---|
| `water_steam` | `mode` = `state`: two quantities among the pairs (`p`, `T`), (`p`, `x`), (`T`, `x`), (`p`, `h`), (`p`, `s`), (`p`, `v`), (`T`, `v`); `rho` instead of `v`. `saturation`: one of `p` and `T`. `rankine`: `p_high`, `p_low`; optional `T_high` (without it: saturated steam), `eta_turbine` and `eta_pump` (default 1), `p_reheat` with `T_reheat`, one of `P_net` and `m_dot` | `state`: `region`, `phase`, `p`, `p_bar`, `T`, `T_C`, `v`, `rho`, `h`, `u`, `s`; in a single phase also `cp`, `cv`, `w` (speed of sound); in a mixture `x`. `saturation`: `T_sat`, `T_C`, `p_sat`, `p_bar`, `v_f`, `v_g`, `rho_f`, `rho_g`, `h_f`, `h_g`, `h_fg`, `u_f`, `u_g`, `s_f`, `s_g`, `s_fg`, `cp_f`, `cp_g`. `rankine`: `w_turbine`, `w_pump`, `w_net`, `q_in`, `q_out`, `efficiency`, `efficiency_carnot`, `back_work_ratio`, `x_turbine_exit`, `T_states`, `p_states`, `h_states`, `s_states`, `x_states`; with `P_net` or `m_dot` also `m_dot`, `P_net`, `Q_in`, `Q_out` |

D51 operations (psychrometrics of humid air, spec 3.19.0). Temperatures in K or
`[value, "C"]`; `RH` as a fraction (0.6) or `[60, "%"]`; `W` in kg/kg or `[10, "g/kg"]`;
pressure `p` (default 101,325 Pa) or `altitude` (ISO 2533 standard atmosphere):

| Operation | Keys of `params` | Keys of `result` |
|---|---|---|
| `psychrometrics` | `mode` = `state`: `T` and one of `RH`, `W`, `T_wb`, `T_dp`, `h`. `mixing`: `T1` and `T2`, for each one of `RH`, `W`, `T_wb`, `T_dp`, `h` with the suffix 1 or 2, dry air flow rates `m1` and `m2` or volume flow rates `V1` and `V2`. `heating_cooling`: state 1 as above, `T2`, optional `RH2` (dehumidifying coil) and `m` or `V1`. `evaporative_cooling`: state 1, optional `eta` (effectiveness, default 1) and `m` or `V1` | `state`: `T`, `T_C`, `p`, `W`, `RH`, `T_dp`, `T_dp_C`, `T_wb`, `T_wb_C`, `h` (J/kg of dry air), `v` (m3/kg of dry air), `rho`, `p_w`, `p_ws`, `mu` (degree of saturation), `q` (specific humidity), `abs_humidity`. In the other modes the same keys with the suffix 1, 2 (and 3 for the mixture); `mixing`: `m_dry_air`; `heating_cooling`: `q_per_kg`, `condensate_per_kg`, with the flow rate `m_dry_air`, `Q`, `condensate`; `evaporative_cooling`: `water_added_per_kg`, with the flow rate `m_dry_air`, `water_added` |

D52 operations (convection coefficient from Nusselt correlations, spec 3.20.0).
Fluid properties given by the caller, at the film temperature: `k`, `Pr`, `nu` or `mu` and
`rho`; `beta` for natural convection (without `beta`, with `T_s` and `T_inf`: `1 / T_film`, ideal gas):

| Operation | Keys of `params` | Keys of `result` |
|---|---|---|
| `convection_correlation` | `geometry` = `flat_plate` (`L`), `cylinder_crossflow`, `sphere_forced` (`D`), `pipe` (`D`, optional `L`, `bc` = `uniform_T` or `uniform_q` in the laminar case, `correlation` = `gnielinski`, `dittus_boelter` with `heating`, `sieder_tate` with `mu` and `mu_w`): one of `v` and `Re`. `vertical_plate` (`L`), `horizontal_plate` (`L` or `A` and `P`, `facing` = `up` or `down`), `horizontal_cylinder`, `sphere_natural` (`D`), `horizontal_enclosure` (`L` = distance): `Ra`, or `dT` or `T_s` and `T_inf` with the properties, optional `g`. Always `k`, `Pr`; optional `A` and `dT` (or `T_s`, `T_inf`) for the heat | `correlation`, `regime`, `Pr`, `Re` or `Gr` and `Ra`, `Nu`, `h`, `L_char`; `beta` if calculated; plate: `h_local_end`; pipe: `f_darcy` (Gnielinski), `viscosity_ratio` (Sieder-Tate); with the temperature difference `q_flux`, with `A` also `Q` |

D53 operations (heat exchangers with the e-NTU method, fins, lumped-parameter transient, spec 3.21.0).
`flow` = `parallel`, `counter`, `crossflow_unmixed`, `crossflow_mixed` (both mixed),
`crossflow_one_mixed` with `mixed` (`cmin` or `cmax` in the modes `effectiveness` and `ntu`; `hot` or `cold`
in the modes `rating` and `sizing`):

| Operation | Keys of `params` | Keys of `result` |
|---|---|---|
| `heat_exchanger_ntu` | `mode` = `effectiveness` (`flow`, `NTU`, `Cr`), `ntu` (`flow`, `epsilon`, `Cr`), `rating` (`flow`, `Th_in`, `Tc_in`, capacities, `UA` or `U` and `A`), `sizing` (`flow`, `Th_in`, `Tc_in`, capacities, one of `Q`, `Th_out`, `Tc_out`, optional `U`). Capacity of each fluid: `C_h` (W/K) or `m_h` and `cp_h`; `C_c` or `m_c` and `cp_c`; `phase_change` = `hot` or `cold` for the fluid that changes phase (infinite capacity, Cr = 0) | `epsilon`, `NTU`, `Cr`; `NTU_alt` (second root, both mixed); with the fluids: `C_h`, `C_c`, `C_min`, `C_max`, `UA`, `A`, `Q_max`, `Q`, `Th_out`, `Tc_out` |
| `fin_heat_transfer` | `tip` = `convective`, `adiabatic`, `fixed_T` (with `T_L`), `infinite` (without `L`); `shape` = `pin` (`D`), `rectangular` (`w`, `t`), `custom` (`P`, `A_c`); `L`, `h`, `k`, `T_b`, `T_inf`; optional `x` (temperature at that point), `n_fins` and `A_base` (array: `A_base` is the wall with the footprints of the fins) | `P`, `A_c`, `m`, `M`, `mL`, `Q_f`, `eps_f`, `A_f`, `eta_f`, `T_tip`, `T_x`; array: `A_t`, `Q_total`, `eta_o` |
| `lumped_transient` | `shape` = `custom` (default, `V` and `A`) or `sphere` (`D`); `rho`, `cp`, `h`, `T_0`, `T_inf`, one of `t` and `T`; optional `k` for the Biot number | `V`, `A`, `L_c`, `tau`, `Bi`, `t`, `T`, `theta_ratio`, `Q` (heat given up by the body, positive if it cools) |

D54 operations (radiation between grey diffuse surfaces, spec 3.22.0):

| Operation | Keys of `params` | Keys of `result` |
|---|---|---|
| `radiation_exchange` | `mode` = `view_factor_2d` with `geometry` = `parallel_plates` (`w_i`, `w_j`, `L`), `inclined_plates` (`w`, `alpha_deg`), `perpendicular_plates` (`w_i`, `w_j`), `three_sided` (`w_i`, `w_j`, `w_k`); `two_surface` with `T1`, `T2`, `eps1`, `eps2` and `config` = `general` (`A1`, `A2`, `F12`), `enclosed` (`A1` inside `A2`, F12 = 1), `parallel_plates` (`A`, optional `n_shields` and `eps_shield`); `enclosure` with the lists `A`, `F` (N x N matrix), `eps`, `T` and `Q` (for each surface one is known and the other `null`) | `view_factor_2d`: `F_ij`, `F_ji`; `two_surface`: `Q` (from 1 to 2), `q1`, `R_total`, `F12`, `F21`, `T_shields`; `enclosure`: lists `J`, `G`, `Q`, `T` |

D55 and D56 operations (electromagnetism, spec 3.23.0 and 3.24.0). Quantities in SI; charges
also accept `mC`, `uC`, `nC`, `pC`; the B field also `mT`, `uT`, `nT`, `G`; vectors and positions are
lists of numbers in SI units:

| Operation | Keys of `params` | Keys of `result` |
|---|---|---|
| `electrostatics` | `mode` = `coulomb` (`q1`, `q2`, `r`, `eps_r`), `point_charges` (`charges` = list of `[q, x, y]` or `[q, x, y, z]`, `point`, optional `q` and `eps_r`), `capacitor` (`geometry` = `parallel_plate` with `A`, `d`; `coaxial` with `R_inner`, `R_outer`, `L`; `eccentric_cylinders` also with `d`; `parallel_wires` and `wire_wall` with `a`, `d`, `L`; `concentric_spheres`; `sphere` and `disc` with `R`; optional `eps_r` and one of `V` and `Q`), `combination` (`values`, `connection` = `series` or `parallel`) | `coulomb`: `k_e`, `F` (positive = repulsion), `E1`, `V1`, `U`; `point_charges`: `E`, `E_mag`, `V`, `U_system`, `F`, `F_mag`; `capacitor`: `C`, `Q`, `V`, `U`, `E` (plane); `combination`: `C_eq` |
| `magnetism` | `mode` = `wire` (`I`, `r`, optional `R_wire`), `loop` (`I`, `R`, optional `x`, `N`), `solenoid` (`N`, `I`, `L`, optional `R` and `z` for the finite solenoid, `mu_r`), `toroid` (`N`, `I`, `r`, `mu_r`), `force_wires` (`I1`, `I2`, `d`, optional `L`), `force_on_wire` (`I`, `L`, `B`, `angle_deg`), `lorentz` (`q`, `v`, optional `E`, `B`, `m`) | `B`, `H`, `n`; `F_per_L`, `F` (positive = attraction); `lorentz`: `F`, `F_mag`, with `m` and `B`: `v_perp`, `r_c`, `T_c`, `f_c` |
| `inductance` | `geometry` = `solenoid` (`N`, `A` or `D`, `L`, `mu_r`), `toroid` (`N`, `h`, `r_inner`, `r_outer`, `mu_r`), `coaxial` (`a`, `b`, `L`, `mu_r`), `straight_wire` (`L`, `r`, `hf`), `circular_loop` (`r`, `a`, `Y`), `rectangle` (`l1`, `l2`, `a`, `Y`), `parallel_wires` (`s`, `a`, `L`, `Y` or `hf`), `wheeler` (`N`, `D`, `L`); optional `I` | `L`; with `I`: `flux_linkage`, `W` |
| `electromagnetic_induction` | `mode` = `flux_change` (`N`, `dt`, and `dPhi` or `A` with `B1` and `B2`), `rotating_coil` (`N`, `A`, `B`, one of `omega` and `rpm`), `motional` (`B`, `L`, `v`) | `emf`, `dPhi`; `emf_peak`, `emf_rms`, `omega`, `f` |

D57 operations (linkages and epicyclic gear trains, spec 3.25.0):

| Operation | Keys of `params` | Keys of `result` |
|---|---|---|
| `linkage` | `mode` = `grashof` (`a` input crank, `b` output, `g` frame, `h` coupler), `four_bar` (the 4 lengths, `theta2_deg`, `assembly` = 1 or -1, optional `omega2`), `slider_crank` (`r` crank, `l` connecting rod, `angle_deg` from top dead centre, optional one of `omega` and `rpm`), `quick_return` (one of `beta_deg` and `Q`) | `grashof`: `T1`, `T2`, `T3`, `S_plus_L`, `P_plus_Q`, `grashof`, `input`, `output`; `four_bar`: `A`, `B`, `theta3_deg`, `theta4_deg`, `transmission_angle_deg`, `omega3`, `omega4`; `slider_crank`: `x`, `dx_dA`, `d2x_dA2`, `omega`, `v`, `a`; `quick_return`: `Q`, `beta_deg` |
| `planetary_gear` | `mode` = `planetary` (`N_s`, `N_r`, two of `omega_s`, `omega_r`, `omega_c`), `differential` (two of `omega_L`, `omega_R`, `omega_c`); speeds in the same unit | the three speeds; `planetary`: also `N_p`, `omega_p` and the ratio with one member fixed (`ratio_s_to_c`, `ratio_r_to_c`, `ratio_s_to_r`) |

D58 operations (catenary cables and journal bearings, spec 3.26.0):

| Operation | Keys of `params` | Keys of `result` |
|---|---|---|
| `cable_catenary` | `mode` = `span_length` (`w`, `H`, `L`, optional `v` height difference), `span_sag` (`w`, `H`, `sag`, supports at the same height), `span_tension` (`w`, `H`, `T0`) | `a`, `T0`, `L`, `sag`, `x_supports`, `y_supports` (heights above the vertex), `T_supports`, `T_max`, `angle_support_deg` |
| `journal_bearing` | `D` or `r`, `c` (radial clearance), `l`, `mu`, `N` (rev/s) or `rpm`, `W` or `P` | `r`, `N`, `omega`, `P`, `hersey`, `S`, `S_omega`, `f`, `T_friction`, `P_loss` |

D59 operations (trusses, spec 3.27.0):

| Operation | Keys of `params` | Keys of `result` |
|---|---|---|
| `truss` | `nodes` ([x, y] or [x, y, z] in m), `members` ([node i, node j], nodes numbered from 1), `supports` ([node, 1 or 0 for each restrained direction]), `loads` ([node, Fx, Fy(, Fz)] in N, optional), `E`, `A` (one value for all members or a list with one value per member) | `forces` (tension positive), `stresses`, `reactions` ([node, Rx, Ry(, Rz)], zero in the free directions), `displacements` (per node), `lengths`, `strain_energy`, `degree`, `determinacy` (`determinate` or `indeterminate`) |

D60 operations (torsion of non-circular sections, spec 3.28.0):

| Operation | `params` keys | `result` keys |
|---|---|---|
| `torsion_section` | `shape` = `circle` (`R` or `D`), `annulus` (`R_e`, `R_i` or `D_e`, `D_i`), `ellipse` (`p`, `q` semi-axes), `equilateral_triangle` (`a`), `rectangle` (`b`, `t`), `thin_open` (`walls` = [[length, thickness], ...]), `thin_closed` (`A_m`, `walls`); optional `T`, `G`, `L` | `J`, `W_t` (= T / tau_max), `beta` (rectangle), `tau_max`, `tau_walls`, `shear_flow` (closed), `GJ`, `k_t`, `twist_rate`, `theta`, `theta_deg` |

D64 operations (geometric and wave optics, spec 4.1.0):

| Operation | `params` keys | `result` keys |
|---|---|---|
| `geometric_optics` | `mode` = `lensmaker` (`n`; `R1` or `C1` = 1/R1, `R2` or `C2`, 0 = flat face; optional `d` thickness default 0, `n_medium` default 1), `mirror` (exactly two of `f` or `R`, `d_o`, `d_i`), `refracting_surface` (`n1`, `n2`, `R`, one of `d_o` and `d_i`), `two_lenses` (`f1`, `f2`, optional `d` default 0) | `lensmaker`: `afocal`, `f`, `P` (= n_medium / f, dioptres), `BFD`, `FFD`, `H1`, `H2`; `mirror`: `f`, `R`, `d_o`, `d_i`, `magnification`, `real_image`, `upright`; `refracting_surface`: `P`, `f_object`, `f_image`, `d_o`, `d_i`, `magnification`; `two_lenses`: `afocal`, `f`, `P`, `BFD`, `FFD`, `angular_magnification` (afocal) |
| `wave_optics` | `mode` = `malus` (`I0`, `theta_deg`, optional `incident` = `polarized` default or `unpolarized`), `brewster` (`n1`, `n2`), `fresnel` (`n1`, `n2`, optional `theta_deg` default 0), `double_slit` (`wavelength`, `d`, optional `m` default 1, `L`), `single_slit` (`wavelength`, `a`, optional `m` default 1, `L`), `grating` (`wavelength`, `d` or `lines_per_mm`, optional `m` default 1, `theta_i_deg` default 0, `N` lines), `thin_film` (`n_film`, `n2`, `wavelength`, optional `n1` default 1, `theta_deg` default 0, `d`), `rayleigh` (`wavelength`, `D`, optional `L`) | `malus`: `I` (unit of `I0`), `ratio`; `brewster`: `theta_B_deg`, `theta_t_deg`, `R_s`, `R_p`; `fresnel`: `theta_t_deg` (`null` in total reflection), `R_s`, `R_p`, `T_s`, `T_p`, `R_unpolarized`, `T_unpolarized`, `total_internal_reflection`, `critical_angle_deg`; `double_slit`: `theta_deg`, `m_max`, with `L` also `y`, `y_small_angle`, `fringe_spacing`; `single_slit`: `theta_deg`, with `L` also `y`, `central_width`; `grating`: `theta_deg`, `d`, `m_min`, `m_max`, with `N` also `resolving_power`, `delta_lambda`; `thin_film`: `half_wave_shift`, `period`, `d_constructive`, `d_destructive` (thinnest film for a reflected maximum and minimum), with `d` also `opd`, `phase_difference_rad`, `reflection` (`constructive`, `destructive`, `partial`); `rayleigh`: `theta_rad`, `theta_deg`, `theta_arcsec`, with `L` also `x` |

Sign conventions of D64: light travels from left to right. Lenses and the refracting surface: a radius is
positive when its centre of curvature is further along the direction of the light (`R1 > 0`, `R2 < 0` is a
biconvex lens); `d_o > 0` real object on the left, `d_i > 0` real image on the right; `H1`, `H2` are measured
from the first and second vertex, positive to the right; `BFD` from the second vertex, `FFD` in front of the
first. Mirror: real is positive, `f = R/2 > 0` for a concave mirror, `d_o` and `d_i` positive in front of the
mirror. Grating: `sin(theta_i) + sin(theta_m) = m lambda / d`, so `m = 0` gives `theta_m = -theta_i`. Thin film:
`n1` incident medium, `n2` substrate, `theta_deg` angle of incidence in `n1`.

D65 operations (practical electronics, spec 4.2.0):

| Operation | `params` keys | `result` keys |
|---|---|---|
| `electronics` | `mode` = `led_resistor` (`V_s`, `V_f`, `I`, optional `n` LEDs in series default 1, `V_switch` default 0), `opamp` (`config` = `inverting`: `R_in`, `R_f`; `non_inverting`: `R_g`, `R_f`; both with optional `V_in`, `A` open-loop gain, `GBW`; `summing`: `R_f`, `inputs` = [[V, R], ...]; `difference`: `R_1`, `R_f`, `R_2`, `R_g`, `V_1`, `V_2`; `integrator` and `differentiator`: `R`, `C`, optional `f`), `wheatstone` (`R_1`, `R_2` adjustable, `R_3`; optional `R_x` with `V_s` and `R_G`, or `V_G` with `V_s`), `timer_555` (`circuit` = `astable`: `R1`, `R2`, `C`; `monostable`: `R`, `C`), `thermistor` (`model` = `beta`: `R0`, `T0`, `B`; `steinhart_hart`: `a`, `b`, `c`; both with one of `T` and `R`; `fit`: `points` = 2 or 3 pairs [T in K, R in ohm]) | `led_resistor`: `R`, `V_R`, `P_R`, `P_led`, `efficiency`; `opamp`: `gain`, `R_input`, `noise_gain`, `V_out`, `gain_finite_A`, `V_out_finite_A`, `bandwidth` (inverting and non-inverting), `V_out` (summing), `V_out`, `gain_V1`, `gain_V2`, `common_mode_gain`, `balanced` (difference), `tau`, `f_unity`, `gain_magnitude`, `phase_deg` (integrator, differentiator); `wheatstone`: `R_x_balance`, `V_G` (node D relative to node B), `R_th`, `I_G`, `R_x`; `timer_555`: `t_high`, `t_low`, `period`, `f`, `duty` or `t_pulse`; `thermistor`: `R`, `T`, `T_C`, or `B`, `R0`, `T0` (2 points), `a`, `b`, `c` (3 points) |

D65 conventions: ideal op amp unless `A` is given; noise gain 1/beta = 1 + R_f/R_in (or R_f/R_g); bandwidth = GBW / noise gain
(one dominant pole). Wheatstone bridge with the names of the source: `R_x` unknown, `R_2` adjustable, node D between
`R_1` (top) and `R_2`, node B between `R_3` (top) and `R_x`; `V_G` with the galvanometer open; `I_G` from Thevenin.
Thermistor temperatures in K or [value, "C"]; `points` in K and ohm.

D66 and D67 operations (friction in simple machines, spec 4.3.0; electric machines, spec 4.4.0):

| Operation | `params` keys | `result` keys |
|---|---|---|
| `friction` | `mode` = `inclined_plane` (`F_w` weight, `theta_deg` slope, `mu`; optional `push` = `parallel` default or `horizontal`), `leadscrew` (`F` load, `d_m` mean diameter, `l` lead, `mu`; optional `alpha_deg` half the thread angle default 0, `mu_c` with `d_c` collar), `rolling_resistance` (`N` normal force with `C_rr`, or with `b` lever arm and `r` wheel radius) | `inclined_plane`: `F_i_up`, `F_i_down` (< 0: a downhill push is needed), `MA_up`, `efficiency_up`, `angle_of_repose_deg`, `self_locking`; `leadscrew`: `T_raise`, `T_lower` (< 0: the load runs back), `lead_angle_deg`, `efficiency`, `self_locking`, with the collar `T_c`, `T_raise_total`; `rolling_resistance`: `F_r`, `C_rr` |
| `electric_machine` | `mode` = `transformer` (`calc` = `ratio` default: `N_p`, `N_s`, optional `V_p`, `I_p`, `Z_s`; `emf`: `f`, `N`, `A` core area, one of `B_peak` and `E_rms`), `dc_motor` (`V_m`, `R_m`, `k` motor constant in N*m/A = V*s/rad, optional `T_load`), `induction_motor` (`f`, `poles`, one of `s` and `n_rpm`; optional circuit per phase `V` line-to-line, `R_s`, `X_s`, `R_r`, `X_r`, `X_m`), `synchronous_machine` (`V`, `E` line-to-line, `X_s`, optional `delta_deg`; and/or `f`, `poles`) | `transformer`: `a` turns ratio, `V_s`, `I_s`, `Z_p`, or `E_rms`, `B_peak`; `dc_motor`: `omega_0`, `n_0_rpm`, `T_stall`, `I_stall`, `P_max`, `T_at_P_max`, with `T_load` also `I_a`, `E_b`, `omega`, `n_rpm`, `T`, `P_out`, `P_in`, `efficiency`; `induction_motor`: `n_s_rpm`, `omega_s`, `s`, `n_rpm`, `f_rotor`, with the circuit `I_stator`, `power_factor`, `P_in`, `P_ag`, `P_cu_rotor`, `P_mech`, `T`, `s_T_max`, `T_max`; `synchronous_machine`: `n_s_rpm`, `omega_s`, `P_max`, `P`, `T_max`, `T` |

D66 and D67 conventions: Coulomb friction against the motion, constant speed. The thread-angle formula of the
leadscrew neglects the effect of the lead angle on the thread angle (warning). Induction motor: star-connected,
per-phase circuit, `V` line-to-line; `poles` is the number of poles (even). Synchronous machine: `P = V E sin(delta) / X_s`
with line-to-line `V` and `E` (= 3 V_ph E_ph sin(delta) / X_s).

D68 and D69 operations (thin plates, spec 4.5.0; continuous beams, spec 4.6.0):

| Operation | `params` keys | `result` keys |
|---|---|---|
| `plate_bending` | `shape` = `circle` (`a` radius) or `rectangle` (`a`, `b` sides); `support` = `clamped` or `simply_supported` (circle), `simply_supported` (rectangle); `h` thickness, `E`, `nu`, `q` uniform pressure | `D`, `w_max` (centre, along q), `M_max`, `sigma_max` (= 6 M / h^2), `sigma_max_at`; circle also `M_center`, `M_edge`; rectangle also `Mx_center`, `My_center`, `sigma_x_center`, `sigma_y_center` (moments per unit width) |
| `continuous_beam` | `spans` = list of `{L, w, P, EI}` (`w` uniform load per metre, `P` = [[force, a from the left end of the span], ...], loads positive downward), `EI` for the spans without their own; optional `settlements` (one per support, m, positive downward), `left` and `right` = `pinned` (default) or `fixed` | `M_supports` (sagging positive), `reactions` (upward positive), `M_span_max` and `x_span_max` (from the left end of each span), `M_max_abs` |

D68 and D69 conventions: Kirchhoff-Love thin plates (small deflection, warning when w_max > 0.2 h). Continuous beams:
Euler-Bernoulli, moments sagging positive; the settlement term uses the sign checked by the stiffness method,
opposite to the formula printed on the source page (see D69).

D70 operation (local losses at sudden changes of section, spec 4.7.0):

| Operation | `params` keys | `result` keys |
|---|---|---|
| `local_loss` | `mode` = `sudden_expansion` or `sudden_contraction`: section 1 (`D1` or `A1`), section 2 (`D2` or `A2`), `Q` or `v1`, `rho`, optional `g`; expansion optional `xi` (loss coefficient, default 1); contraction optional `mu` (contraction coefficient, default Weisbach). `mode` = `borda_carnot`: `xi`, `v1`, `v2`, `rho`, optional `g` | `dE` (energy loss per unit volume, Pa), `dH` (head loss), `dp` = p1 - p2 and `dh` (horizontal pipe), `K_1` and `K_2` (loss coefficients referred to v1 and v2), `A1`, `A2`, `Q`, `v1`, `v2`, `xi`; contraction also `mu`, `A3`, `v3` (vena contracta) |

D70 conventions: Borda-Carnot loss Delta E = xi (rho / 2) (v1 - v2)^2, zero with a warning when v1 <= v2.
Weisbach contraction coefficient mu = 0.63 + 0.37 (A2/A1)^3 for a sharp edge. dp = Delta E + (rho / 2) (v2^2 - v1^2)
from the energy balance (no change of elevation).

D71 operation (stress concentration at holes in an infinite plate, spec 4.8.0):

| Operation | `params` keys | `result` keys |
|---|---|---|
| `stress_concentration` | `mode` = `circular_hole`: `sigma_x` (far-field stress along x), optional `sigma_y`, `a` (hole radius), optional point `r` (>= a) and `theta_deg` (from x). `mode` = `elliptical_hole`: `sigma` (far-field stress), `a` (semi-axis perpendicular to the stress) and one of `b` (other semi-axis) or `rho` (radius of curvature at the end of a) | circular: `sigma_edge_max`, `theta_edge_max_deg`, `sigma_edge_min`, `theta_edge_min_deg`, `Kt`; with a point also `sigma_rr`, `sigma_tt`, `sigma_rt`. Elliptical: `Kt`, `sigma_max`, `b`, `rho` |

D71 conventions: Kirsch field for sigma_x, plus the same field turned by 90 degrees for sigma_y (superposition).
K_t = largest edge stress magnitude / largest applied stress magnitude: 3 uniaxial, 2 equal biaxial, 4 pure shear.
Inglis: K_t = 1 + 2 a/b = 1 + 2 sqrt(a/rho); not usable for a crack (rho -> 0).

D72 operation (unbalance and balancing of rigid rotors, spec 4.9.0):

| Operation | `params` keys | `result` keys |
|---|---|---|
| `rotor_balance` | `mode` = `unbalance`: `masses` = list of `{m, r, angle_deg, z}` (z axial position, default 0), optional `M` (rotor mass), `n_rpm` or `omega`, `planes` = [z_L, z_R], `r_correction` (one radius or [r_L, r_R]). `mode` = `iso_grade`: `G` (balance grade in mm/s), `M`, `n_rpm` or `omega` | unbalance: `U`, `U_angle_deg`, `C` and `C_angle_deg` (couple about z = 0), `U_correction`, `U_correction_angle_deg`, optional `e`, `F`, `M_couple`, `m_correction`; with planes `U_L`, `U_L_angle_deg`, `U_R`, `U_R_angle_deg`, optional `m_L`, `m_R`. ISO grade: `e_per`, `U_per`, `U_per_g_mm` |

D72 conventions: unbalances as complex numbers m r e^(i theta) in the rotor frame; two-plane correction from
U_L + U_R = -S and z_L U_L + z_R U_R = -C (no resultant centrifugal force or couple). ISO grade: e_per omega = G.

D73 operation (rolling without slipping of a rigid body, spec 4.10.0):

| Operation | `params` keys | `result` keys |
|---|---|---|
| `rolling` | `mode` = `incline` (`theta_deg`) or `force` (`F` through the centre, level surface); `m`, `r`, one of `I` (about the centre) or `r_g` (radius of gyration), optional `g`, `mu_s` (static friction coefficient), `v` | `a`, `alpha`, `F_net`, `f` (friction at the contact), `N`, `I_over_m_r2`, `mu_min`; with `mu_s` `rolls_without_slipping`; with `v` `K_translation`, `K_rotation`, `K` |

D73 conventions: a = F_along / (m + I / r^2); friction f = F_along - m a; pure rolling while |f| <= mu_s N,
otherwise a warning (the slipping motion is not computed).

D74 (spec 4.10.1): `eng_calc` never answers ok with a non-finite number. An overflow or underflow of the inputs
gives `MATH_ERROR` ("The result is not finite").

D87 operation (kinetic theory of gases, spec 4.11.0):

| Operation | `params` keys | `result` keys |
|---|---|---|
| `kinetic_theory` | `mode` = `speeds` (`T`; optional `dof`), `pressure` (`rho`, `v_rms`), `distribution` (`T`; `v` and/or `v1` < `v2`), `mean_free_path` (`T`, `p`, one of `d` kinetic diameter, `sigma` cross-section, `mu` dynamic viscosity; optional `L`), `wall_flux` (`T`, `p`; optional `A` hole area), `graham` (`M1` and one of `M2`, `rate_ratio` = rate_1 / rate_2), `equipartition` (one of `gas` = `monatomic` / `diatomic` / `polyatomic` or `dof`; optional `T`, `n_mol`). The gas is one of `m` (mass of one molecule, also `[28, "u"]`) or `M_molar`; it is optional in `pressure` and `mean_free_path` (with `mu` it is needed) | `speeds`: `m`, `M_molar`, `v_p`, `v_mean`, `v_rms`, `v_rel`, `E_k`, `E_k_molar`, with `dof` `gamma`, `c_sound`; `pressure`: `p`, `K_t_density`, with the gas `T`, `n`; `distribution`: `a`, `v_p`, with `v` `pdf`, `cdf`, `fraction_above`, with `v1`, `v2` `fraction_between`; `mean_free_path`: `n`, `sigma`, `ell`, with the gas `v_mean`, `z`, `tau`, with `L` `Kn`; `wall_flux`: `n`, `v_mean`, `J`, with `A` `Phi`, `mdot`; `graham`: `M1`, `M2`, `rate_ratio`, `time_ratio`; `equipartition`: `dof`, `Cv_molar`, `Cp_molar`, `gamma`, with `T` `U_molar`, `E_molecule`, with `n_mol` `C_V`, `C_p` (and `U`) |

D87 conventions: Maxwell-Boltzmann speeds of an ideal gas in equilibrium; `v_rel` = sqrt(2) `v_mean`; mean free
path l = 1 / (sqrt 2 n sigma), sigma = pi d^2, or (mu / p) sqrt(pi k T / (2 m)) for hard spheres with the same
viscosity; z = `v_mean` / l; wall flux J = p / sqrt(2 pi m k T); the fraction above v is computed without
cancellation in the tail. Equipartition is classical: 1/2 k T per degree of freedom, vibrational modes only
when counted in `dof`.

D88 operation (Newtonian gravitation, spec 4.12.0):

| Operation | `params` keys | `result` keys |
|---|---|---|
| `gravitation` | `mode` = `two_bodies` (`m1`, `m2`, `r`), `point_masses` (`masses` = list of `[m, x, y, z]` in kg and m, at most 1000; `point` = `[x, y, z]`; optional `m_test`), `sphere` (one of `M`, `mu`, `rho`; `R`, `r` from the centre; optional `shell` = true for a hollow thin shell, `m_test`), `ring` (one of `M`, `mu`; `R`, `z` on the axis), `rotating_body` (one of `M`, `mu`; `R`; one of `omega`, `T_rot`; `latitude_deg`), `gauss` (one of `M`, `Phi_G`), `potential_difference` (one of `M`, `mu`; `r1`, `r2`; optional `m`) | `two_bodies`: `F`, `U`, `g1`, `g2`, `mu1`, `mu2`; `point_masses`: `g` (vector), `g_magnitude`, `potential`, `M_total`, `center_of_mass`, with `m_test` `F` (vector), `F_magnitude`, `U`; `sphere`: `mu`, `g`, `potential`, `g_surface`, `potential_surface`, with the mass `rho` and (solid) `binding_energy`, `binding_energy_per_kg`, with `m_test` `F`, `U`; `ring`: `mu`, `g`, `potential`, `z_max`, `g_max`; `rotating_body`: `omega`, `g_grav`, `a_centrifugal`, `g_radial`, `g_tangential` (towards the equator, >= 0 in both hemispheres since 4.12.1), `g_eff`, `deflection_deg` (>= 0, towards the equator); `gauss`: `Phi_G`, `M_enclosed`; `potential_difference`: `dPhi`, `g1`, `uniform_field_error`, with `m` `W`, `W_uniform` |

D88 conventions: `g` in `point_masses` is a vector pointing towards the masses; elsewhere `g` is the magnitude
towards the centre (or the plane of the ring). Potentials are 0 at infinity. A uniform solid sphere has
g = G M r / R^3 inside; a hollow shell has g = 0 inside. The rotating body is a sphere: the centrifugal
acceleration omega^2 R cos(latitude) points away from the axis (no flattening, no J2).

Corrections of the audit of 2026-11-25 (spec 4.13.0, decisions D90-D97):

- `math_solve` and `math_verify`: `verified` (and `valid`) is true only after a real check. `diff` and
  `partial_diff` compare with a 40-digit numerical derivative; systems of ODEs are checked by substitution;
  `polynomial` nroots, discriminant, resultant, vieta, coeffs and degree have independent numerical checks;
  an empty list of solutions is not verified. A residual that is undefined (`zoo`, `nan`) is not zero.
- `math_eval`: `poisson_pmf`, `poisson_cdf`, `binom_pmf`, `binom_cdf`, `gcd`, `lcm`, `mod_inv` and the decimals
  of `round` need integer counts (INVALID_ARGUMENT otherwise); Poisson mean >= 0, 0 <= p <= 1; outside 0..n
  the probability is 0; the CDFs take the floor of k.
- `math_numeric.quad` and `math_solve.multi_integrate`: the first entry of `bounds` is the innermost integral
  and a bound may use only the variables of the outer entries (INVALID_ARGUMENT otherwise).
- `eng_calc`: `gear_spur` 0 < `alpha_deg` < 45; `belt_drive` `m_l` needs `P` and `n`; `bolt_joint` uses the
  magnitude of `Fv` and needs `Ft` >= 0; `psychrometrics` heating_cooling `RH2` needs `T2` < `T1`; the
  thermistor never answers T <= 0 K; `control_system` second_order times are exact for any damping.
  `kinetic_theory` and `gravitation` give the D74 error for inputs outside the floating-point range.
- `geom3d_calc`: counts (`n`, `count`, `n_holes`) are integers, checked before any list is built;
  `mesh_properties` of a mesh with volume 0 has `centroid` and `inertia` null; a parametric curve with
  non-finite points is INVALID_ARGUMENT, a divergent length MATH_ERROR; any other curve whose points or
  length leave the floating-point range gives the D74 MATH_ERROR.
- `python_sandbox`: numpy integers count for the caps; the shift cap is on the size of the result.

Conventions:

- `beam_calc`: loads downward, reactions upward, moments and deflections in absolute value,
  `x` from the left end; the cantilever has the fixed end at `x = 0`. Formulas: Euler-Bernoulli, closed
  handbook cases.
- `shaft_torsion_bending`: stresses on the outer surface; `sigma_eq = sqrt(sigma_b^2 + 3 tau^2)`
  (Von Mises) or `sqrt(sigma_b^2 + 4 tau^2)` (Tresca).
- `euler_buckling`: with `A` and `Sy`, if the slenderness is below `sqrt(2 pi^2 E / Sy)` the Johnson
  parabola applies (`method: "johnson"`) and there is a warning.
- `pressure_vessel`: thin wall with the **mean diameter** `d_m = d_i + t` and the net pressure
  `p - p_o`; the design values are the Lamé ones (thick wall, exact); warning if
  `t / r_i > 0.1`. Von Mises on the inner surface.
- `kinematics_dynamics`: with `mode` only the keys of the mode apply; `projectile` is without
  air resistance (with air: `math_numeric.ode_ivp` with `events`, section 6.1).
- `vibration_1dof`: forcing `F0 cos(omega t)`; `phase_deg` is the lag of the displacement.
- `belt_drive`: open belt; `n` is of the small pulley; `T1`, `T2` with the centrifugal
  force `m_l v^2` if `m_l` is present.
- `cnc_machining`: `kc` in Pa (`[2000, "MPa"]` = 2000 N/mm2); `P_cut = kc MRR`.
- `gear_spur`: normal teeth (addendum `m`, dedendum `1.25 m`), without profile shift;
  transverse `contact_ratio`; forces on the pinion without losses.
- `fluid_pipe`: laminar below Re = 2300 (`f = 64 / Re`), transition up to 4000 with a warning,
  then turbulent; `f` from Colebrook-White solved with `brentq`; `p2` at constant section (Bernoulli
  with the losses); without `L` no distributed losses.
- `aero_hydro_force`: with `T` the speed of sound is that of dry air (ISO 2533: gamma = 1.4,
  R = 287.05287 J/(kg K)); `v_terminal` without Archimedes buoyancy, g = 9.80665 m/s2.
- `heat_transfer`: temperatures in K as a number or `[value, "C"]`, `"K"`, `"F"`; below absolute
  zero `INVALID_ARGUMENT`. `radiation`: small grey body in a large environment, sigma =
  5.670374419e-8 W/(m2 K4) (CODATA 2018). `lmtd`: `Q = U A LMTD`; in parallel flow the cold outlet
  cannot exceed the hot outlet.
- `heatsink_thermal`: series chain `T_j = T_a + P (R_jc + R_cs + R_sa)`; `R_sa_required <= 0`
  gives `MATH_ERROR` (no heat sink is enough).
- `circuit_ac_dc`: in three-phase `V` is the line-to-line voltage; inductive load (`phi >= 0`).
  `cable_drop` uses only the conductor resistance at 20 degrees (warning), IEC 60028 copper =
  1/58 ohm mm2/m, `dV = k R I cos(phi)` with k = 2 in single-phase and sqrt(3) in three-phase.
- `rlc_filter`: output without load; in the band-pass the output is on the resistor.
- `foc_clarke_park`: `amplitude` (factor 2/3) preserves amplitudes, `power` (factor sqrt(2/3))
  preserves power; with `theta_deg = 0` the d axis coincides with phase a.
- `control_system`: `second_order` is the unit step response of
  `wn^2 / (s^2 + 2 zeta wn s + wn^2)`; `ts_2pct` and `tr_10_90` are calculated on the exact response,
  `ts_2pct_approx = 4 / (zeta wn)` is the short formula. `margins` uses the first crossing.
  `pid_zn` uses the original Ziegler and Nichols tables (1942).
- `optics_acoustics`: angles in degrees; negative `d_i` = virtual image; reference pressure
  20 micropascal (ISO 1683); `sum_levels` for incoherent sources; `inverse_square` in
  free field.
- `hydrostatics`: without `p0` the pressure is relative to the free surface; without `g` it is 9.80665 m/s2
  (also in the other operations of areas E and F). `plane_wall`: force on the wetted face,
  without atmospheric pressure (it acts on both faces); `y` is measured along the wall
  from the free-surface line; `cp_offset = y_cp - y_c = I_xc / (y_c A)`.
- `pump_system`: `affinity` holds for geometrically similar pumps (`Q` proportional to `n D^3`,
  `H` to `n^2 D^2`, `P` to `rho n^3 D^5`); with `D1` different from `D2` there is a warning, because the
  trimming of the impeller of the same pump follows these laws only approximately.
  `npsh`: `NPSH_a = (p_s - p_v) / (rho g) + z_s - h_loss`; negative margin = cavitation, with a
  warning. `cylinder`: theoretical forces multiplied by `eta`, without back pressure.
- `open_channel`: Manning formula in SI, `v = R_h^(2/3) S^(1/2) / n`, uniform flow; critical
  depth from `Q^2 T / (g A^3) = 1`; in the circular pipe the maximum flow rate is reached before the
  pipe is full and a `Q` above `Q_max` gives `MATH_ERROR`.
- `flow_meter`: `Q = Cd eps A_throat sqrt(2 dp / rho) / sqrt(1 - beta^4)`; `Cd` is given by the
  manufacturer or by the ISO 5167 standard, the server does not estimate it. Sharp-edged rectangular weir
  `Q = Cd (2/3) b sqrt(2 g) h^(3/2)`; triangular `Q = Cd (8/15) tan(theta/2) sqrt(2 g) h^(5/2)`.
- `compressible_flow`: perfect gas with constant `gamma`, adiabatic flow; default `R`
  of dry air (ISO 2533); `normal_shock` with the Rankine-Hugoniot relations; `choked` is the
  maximum flow rate, with M = 1 in the throat.
- `ideal_gas`: universal constant `R_u = 8.314462618 J/(mol K)` (CODATA 2018, exact in SI);
  `W` is the work done by the gas, positive in expansion; `Q = dU + W`; `cv = R / (gamma - 1)` and
  `cp = gamma R / (gamma - 1)` constant; `polytropic` with `n_poly = 1` is the isotherm.
- `thermo_cycle`: ideal air-standard cycles with constant specific heats, states numbered from
  1 (start of compression); `q_in`, `q_out`, `w_net` in J/kg. The Rankine cycle needs the
  steam tables and is not here.
- `calorimetry`: `mixture` without phase changes and without exchange with the outside; `dT` is a
  difference in K (a number, without conversion from Celsius).
- `fatigue_life` (D44): Shigley method. `Se_prime = 0.5 Sut` up to 1400 MPa, then 700 MPa;
  `ka = a Sut^b` (MPa) with a, b per finish; `kb = 1.24 d^-0.107` (d in mm, 2.79-51 mm; axial
  `kb = 1`); `kc` 1 bending, 0.85 axial, 0.59 torsion; `ke` from the reliability table.
  Gerber solves `n sa/Se + (n sm/Sut)^2 = 1`. `sn_life`: Basquin line between `(10^3, f Sut)` and
  `(10^6, Se)`, `a = (f Sut)^2 / Se`, `b = -(1/3) log10(f Sut / Se)`, `N = (sa / a)^(1/b)`; `f` has
  no default value. `ec3`: EN 1993-1-9 curves with `delta_sigma_D = (2/5)^(1/3) delta_sigma_C`
  at 5e6 cycles (m = 3 before, m = 5 after) and `delta_sigma_L = (5/100)^(1/5) delta_sigma_D` at 1e8;
  shear m = 5 with `delta_tau_L = (2/100)^(1/5) delta_tau_C`. `miner`: damage `D = sum(n/N)`,
  `ok` if `D <= 1`.
- `bolt_joint` (D44): `d2 = d - 0.649519 P`, `d3 = d - 1.226869 P`, `As = pi/4 ((d2 + d3)/2)^2`
  (ISO 898-1); nominal `fyb`, `fub` from the grade (EN 1993-1-8 table 3.1, 12.9 with the same
  rule as the name and with a warning); EN 1993-1-8 table 3.4: `Fv_Rd = alpha_v fub A / gamma_M2` per
  shear plane (`alpha_v` 0.6 or 0.5; `A = As` if the thread is in the plane, otherwise the area of the
  shank), `Ft_Rd = k2 fub As / gamma_M2` (`k2` 0.9, countersunk 0.63),
  `Fv/Fv_Rd + Ft/(1.4 Ft_Rd) <= 1`, `Fp_C = 0.7 fub As`, `Fs_Rd = ks n mu (Fp_C - 0.8 Ft)/gamma_M3`.
  Torque: `T_K = K d Fp_C`; VDI 2230: `T = Fp_C (0.159 P + 0.578 d2 mu_thread + D_km mu_head / 2)`.
  The bearing resistance of the plate is not here (hole distances are needed).
- `weld_fillet` (D44): EN 1993-1-8 4.5.3; `a = z / sqrt(2)`; simplified
  `fvw_d = fu / (sqrt(3) beta_w gamma_M2)`, `Fw_Rd = fvw_d a`; directional
  `sqrt(sigma_perp^2 + 3 (tau_perp^2 + tau_par^2)) <= fu / (beta_w gamma_M2)` and
  `sigma_perp <= 0.9 fu / gamma_M2`. `fu` (EN 10025-2 for S235-S355, t <= 40 mm or 40-80 mm;
  S420 and S460 only t <= 40 mm) and `beta_w` (0.8; 0.85; 0.9; 1.0; 1.0) of the weaker steel.
- `material_lookup` (D44): values taken from cited sources for each property, also ranges; a
  property with no source found is `null`, never estimated.
- Modes of D45: `collision` is a central collision in one dimension, signed velocities, impulse
  `J = m1 (v1_after - v1)`; `orbit` uses `G = 6.67430e-11 m3/(kg s2)` (CODATA 2018) with `M`,
  Kepler's law `T = 2 pi sqrt(a^3 / mu)` and the vis-viva equation `v^2 = mu (2/r - 1/a)`; `gyroscope`
  `T = I omega Omega_p` (slow precession); `flywheel` `Cs = (omega_max - omega_min) / omega_mean`
  with `omega_mean` the arithmetic mean and `dE = I Cs omega_mean^2`; disk `I = m R^2 / 2`, thin ring
  `I = m R^2`. `doppler` holds for waves in a medium at rest, `|v_source| < c`; `string` and `pipe` without
  end correction; `intensity` in free field with references 1 pW/m2 and 1 pW (ISO 1683).
  `impedance` with reactances `X_L = 2 pi f L`, `X_C = -1 / (2 pi f C)`; `transient` with the circuit at
  rest before the step (`charge`) or the capacitor charged to `V` (`discharge`).
- D46 operations: `thermal_stress` linear (`dL = alpha L dT`); `bar` is a bar with fixed
  ends (`sigma = -E alpha dT`) or between two rigid walls with a gap (stress only
  if `alpha L dT > gap`); `plate` is a plate restrained in both in-plane directions
  (`sigma = -E alpha dT / (1 - nu)`). `alpha` accepts `1/K`, `um/(m*K)`, `ppm/K`. `contact_hertz`
  holds for elastic, smooth, non-conforming bodies (contact area small compared with the radii), with
  `1/E* = (1 - nu1^2)/E1 + (1 - nu2^2)/E2`; spheres `a = (3 F R / (4 E*))^(1/3)`,
  `p0 = 3 F / (2 pi a^2)`, `d = a^2 / R`; cylinders `b = sqrt(4 F R / (pi L E*))`,
  `p0 = 2 F / (pi b L)`; `tau_max` is the numerical maximum of the principal shear stress
  on the axis (Johnson stresses 1985). `spring_helical`: `k = G d^4 / (8 D^3 Na)`, `tau` with the
  Bergsträsser factor (Shigley), `tau_wahl` with the Wahl one; total coils and solid
  length according to the end type; `f_n = (1/2) sqrt(k / m_active)` with the spring between two parallel
  planes; warning if the index `C` falls outside 4-12. `interference_fit`: Lamé thick cylinders in
  plane stress, hub and shaft of the same length, `delta` = shaft diameter minus
  bore diameter; `T = mu p pi d^2 L / 2`, `F_axial = mu p pi d L`;
  `dT_assembly = (delta + clearance) / (alpha_hub d)`. `strain_analysis`: plane strain with
  engineering shear strain `gxy`; `theta_p_deg` from `x` towards the direction of `e1`; rosette
  readings `e(t) = ex cos^2 t + ey sin^2 t + gxy sin t cos t`, three different angles less than 180
  degrees apart; stresses in plane stress.
- D48 operations: **abridged** CIAAW/IUPAC atomic weights, downloaded on 2026-11-24 (`data/atomic_weights.json`,
  118 elements, 84 with an abridged weight; for elements without stable isotopes an error), uncertainty
  of the molar mass as the sum of the uncertainties of the abridged weights. The balancing is the smallest
  integer vector of the kernel of the element-by-species matrix (SymPy); more than one kernel vector or coefficients
  not all positive are an error. `acid_base` solves the exact charge balance (cubic for
  weak acids and bases and for the buffer, quadratic for the strong ones) with the autoionisation of water, and
  also gives the textbook approximation (`pH_approx`, `pH_henderson`). Kinetics: integrated laws of
  order 0, 1, 2 with a single reactant; Arrhenius with CODATA `R`. Electrochemistry: Nernst with the natural
  logarithm, Faraday `n = Q / (z F)`, `dG = -n F E = -R T ln K`; `F` and `R` CODATA 2022.
- D60 operations: formulas from the Wikipedia page «Torsion (mechanics)» (table of sections: exact
  solutions for circle, annulus, ellipse and equilateral triangle; thin-strip sum; Bredt). The rectangle
  uses the Fourier series of the Prandtl stress function written here (the source says alpha comes «from a
  Fourier-series solution»), checked in the tests against finite differences and against the beta table of
  «Torsion constant». Derived here: triangle tau_max = 20 T / a^3 from the cubic stress function, wall
  stresses of open sections tau_i = T t_i / J.
- D59 operations: stiffness method taken from the Wikipedia page «Direct stiffness method»
  (member matrix `(E A / L) [c c^T, -c c^T; -c c^T, c c^T]` with the direction cosines `c`, assembly,
  solution on the free degrees of freedom). Pin-jointed members, loads at the nodes, small displacements (warning).
  Degree of static indeterminacy `m + r - d n` (d = 2 or 3); unstable structure = singular reduced matrix, error.
  Frames with bending are not here: the source writes their matrix only with symbols.
- D58 operations: formulas taken from the Wikipedia pages «Catenary» and «Sommerfeld number».
  Catenary: flexible, inextensible cable with uniform weight `w` per metre; `a` found with brentq
  from the relations of the source; for supports at different heights the vertex comes from
  `v = 2 a sinh((x1 + x2) / (2a)) sinh(H / (2a))` (calculated here), with a warning if it falls outside the
  span. Bearing: `S` with N in revolutions per second (with the angular speed `S_omega` = 2 pi S, as
  the source says), Petroff law for the centred, lightly loaded journal (warning).
- D57 operations: formulas taken from the Wikipedia pages «Four-bar linkage», «Piston
  motion equations» and «Epicyclic gearing». Classification of the four-bar linkage from the signs of T1, T2, T3
  (table of the source); with a zero term there is a warning (change point). Calculated by the server and
  indicated in the field `formula`: position of the four-bar linkage as the intersection of two circles (frame from
  (0, 0) to (g, 0); `assembly` chooses one of the two solutions), velocities from the differentiated
  closure equation (2x2 system), acute transmission angle between coupler and rocker, planet with
  `N_p = (N_r - N_s) / 2` (error if not an integer) and `N_p (w_p - w_c) = -N_s (w_s - w_c)`. Worm
  gears and bevel gears are not here: the pages of the source give no formulas.
- D55 and D56 operations: formulas taken from the Wikipedia pages «Coulomb's law»,
  «Capacitance», «Magnetic field», «Biot–Savart law», «Solenoid», «Ampère's force law», «Lorentz
  force», «Inductance» and «Faraday's law of induction»; CODATA 2022 constants from scipy (eps0, mu0).
  Homogeneous medium with `eps = eps_r eps0` and `mu = mu_r mu0`. The server states in the field `formula` the
  relations derived and not written in the source: cyclotron radius and period (Lorentz force and
  circular motion, non-relativistic, with a warning), inductance of the toroid with rectangular section
  (integrated flux of the toroid field), rotating coil and moving bar (Faraday's law). In the
  thin-wire formulas `Y` is 1 (uniform current) if not given, with a warning; `hf = true`
  uses the high-frequency forms of the source. Wheeler: error within 1 % for `l > 0.4 D` (warning).
- D54 operations: formulas taken from the Wikipedia pages «View factor», «Thermal
  radiation» and «Radiosity (radiometry)»; sigma = 5.670374419e-8 W/(m2 K4). View factors only
  for the infinitely long 2D geometries of the source (the widths act as areas in the
  reciprocity); the 3D ones are not here (the source refers to external tables). Two surfaces:
  formula of the cavity of two grey surfaces; with `general` it holds if 1 and 2 alone form the
  cavity (F11 = 1 - F12, warning), and the server rejects A1 F12 > A2. Shields between parallel plates:
  series resistances of the circuit of the source, `1/e1 + 1/e2 - 1 + N (2/e_s - 1)` per unit area,
  with the temperatures of the shields. Cavity of N surfaces: linear system of the radiosities; the server
  checks the summation rule (rows of F = 1 within 1e-6) and the reciprocity (within 1e-6 relative)
  and asks for at least one known temperature.
- D53 operations: formulas taken from the Wikipedia pages «NTU method», «Fin (extended
  surface)» and «Biot number». e-NTU: `Cr = Cmin / Cmax`, `NTU = U A / Cmin`,
  `Q = e Cmin (Th_in - Tc_in)`; with `Cr = 0` (one fluid changes phase) `e = 1 - exp(-NTU)` for every
  flow. Counterflow calculated in the stable form near `Cr = 1`. Cross flow with
  both unmixed: series of the polynomials `P_n` summed in logarithms, `NTU` up to 500. The
  inverses (NTU from effectiveness) are the formulas of the source inverted with algebra; for both
  unmixed and both mixed with brentq. With both mixed the effectiveness has a maximum:
  above the limit `1 / (1 + Cr)` there are two NTU, the server gives the smaller one in `NTU` and the other in
  `NTU_alt`, with a warning. The LMTD correction factor F and shell-and-tube exchangers are not here
  (no verified source). Constant cross-section fins: `m = sqrt(h P / (k A_c))`, the 4 tip conditions
  of the source (cases A-D), calculated without overflow for large `mL`; fin surface
  `A_f = P L`, plus `A_c` with the convective tip; overall efficiency
  `eta_o = Q_total / (h A_t theta_b)`. Transient: `L_c = V / A`, `Bi = h L_c / k`,
  `tau = rho cp V / (h A)`; warning above `Bi = 0.1` and without `k`.
- D52 operations: correlations taken from the Wikipedia pages «Nusselt number»,
  «Churchill–Bernstein equation», «Heat transfer coefficient» and «Reynolds number», which cite
  Incropera and the original papers. Laminar flat plate `Nu_L = 0.664 Re^1/2 Pr^1/3` with a
  warning above `Re = 5e5` (transition); the turbulent moving plate is not here (no verified
  source). Cylinder: Churchill-Bernstein; forced sphere: Faeth (`Re < 1800`); developed laminar
  pipe 3.66 or 4.36 below `Re = 2300`, then Gnielinski with the Filonenko `f` (default),
  Dittus-Boelter (`n` = 0.4 if the fluid is heated, deduced from `T_s > T_inf` if `heating` is missing),
  Sieder-Tate. Vertical wall: Churchill-Chu laminar below `Ra = 1e9`, the general form
  above. Horizontal plate: `L = A / P`, values from Incropera p. 577 according to the hot side.
  Horizontal cylinder: Churchill-Chu; natural sphere: Yuge (Pr about 1); horizontal layer
  heated from below: Globe-Dropkin. Outside the validity range there is a warning, the calculation goes on.
- D51 operations: ideal-gas model of ASHRAE Handbook - Fundamentals (2017) ch. 1,
  written from scratch: `W = 0.621945 p_w / (p - p_w)`, `h = 1.006 t + W (2501 + 1.86 t)` kJ/kg of
  dry air, `v = 287.042 T (1 + W / 0.621945) / p`. The saturation pressure p_ws comes
  from Eq. 30 of IAPWS-IF97, not from the Hyland-Wexler formula: measured deviation between the two at
  most 0.0176 % from 0 to 100 °C. Thermodynamic wet bulb from the adiabatic saturation balance
  with liquid water `h_w = 4.186 t*`, solved with `brentq`; dew point with
  Eq. 31 of IF97. Range from 0 °C to 200 °C: below 0 °C the saturation over ice would be needed,
  which is not here (dew point and wet bulb below 0 °C give `null` with a warning). Enhancement factor
  neglected, as in ASHRAE. Cooling coil: if T2 is below the dew point the outlet is saturated (or has
  `RH2`), the heat includes the condensate that leaves at T2; negative `Q` = heat removed.
  Evaporative cooling at constant wet bulb, `t2 = t1 - eta (t1 - t_wb)`. States beyond
  saturation (fog) give `INVALID_ARGUMENT`.
- D50 operations: IAPWS-IF97 (IAPWS R7-97(2012)); coefficients of tables 1, 2, 10, 11,
  30, 34, 37 and 38 in `tools/if97_data.py`, generated from the official PDF (exponents read from the glyphs).
  Validity: 273.15 K <= T <= 1,073.15 K with p <= 100 MPa; up to 2,273.15 K with p <= 50 MPa;
  outside `INVALID_ARGUMENT` (below 0 °C there is ice). Region from (p, T) with p_s (Eq. 30) and the
  B23 boundary (Eq. 5). Density of region 3 with `brentq` on the isotherm: densest root for the
  liquid, least dense for the vapour. Saturation in region 3 with p_s of Eq. 30 and the two
  roots of Eq. 28 (like Table 44 of the document). Inputs (p, h), (p, s), (p, v), (T, v)
  inverted with `brentq` on the basic equations, without the backward equations; mixture if the
  value falls between saturated liquid and vapour (relative margin 1e-12, the results have 15 digits).
  On the saturation line (p, T) gives the vapour, with a warning within 0.01 %. Zero of internal
  energy and entropy: saturated liquid at the triple point (IF97 convention). Rankine cycle: 1
  saturated liquid at `p_low`, 2 pump outlet, 3 boiler outlet, turbine with isentropic
  efficiency, optional reheat (states 4 and 5); `efficiency_carnot` between the
  condenser temperature and the highest of the cycle; warning if the exit quality is below
  0.88. For ideal-gas cycles `thermo_cycle` remains. Not included: viscosity and thermal
  conductivity (other IAPWS releases), metastable states and ice.
- D49 operations: data from AME2020 and NUBASE2020 (`data/nuclides.json.gz`, 5,383 nuclides:
  3,558 ground states and 1,825 isomers; CC BY 3.0 licence of the papers, attribution in
  `NOTICE`). Atomic mass excesses: AME2020 for ground states, ground state plus
  excitation energy for isomers. Binding energy `B = Z Delta(1H) + N Delta(n) - Delta`
  (atomic masses, as in AME2020). Q and thresholds with nuclear masses `M_atom - Z m_e`; electrons and
  positrons counted explicitly; the binding of the electrons is neglected (for 4 p -> alpha
  it is 24.6 eV out of 24.687 MeV). Uncertainty of Q as a quadratic sum without correlations. Relativistic
  threshold with the target at rest; two-body decays with relativistic energies.
  The years of NUBASE2020 are tropical (365.2422 d); the years written in the parameters (`y`, `a`,
  `ky`, `My`) are Julian (365.25 d): difference 2.1e-5 relative. Two-member chains with
  Bateman; `secular` equilibrium if `lambda1 / lambda2 < 0.01`. Modern physics with the CODATA 2022
  constants and exact relativistic formulas; hydrogen-like with a nucleus of infinite mass, or
  with the reduced mass if `nuclide` is present; wavelengths in vacuum. Black-body fraction in
  a band with the dimensionless Planck integral (`quad`).

Verification of the formulas (D39): every operation is tested with an independent method. Beams against
SymPy `Beam` (singularity functions) on 8 cases; Von Mises from the eigenvalues against the formula
by components; Lamé with the equilibrium of half a tube and half a sphere; minimum shaft diameter
recomputing the stress; K of the fixed-pinned case from the root of `tan(x) = x`. Area B:
projectile and vibrations against the numerical integration of the equations of motion (`solve_ivp` with
impact event, damping from the peaks, steady-state amplitude); belt length from the tangent
points; contact ratio from the intersections of the line of action with the addendum
circles; bearing life with the round trip of `C_required`. Area C: laminar losses
against Hagen-Poiseuille; Colebrook `f` against the explicit formulas of Haaland and
Swamee-Jain (within 2 %); exchanger area against the numerical integration of the balance
along the area, in parallel flow and in counterflow; layered wall with series resistances.
Area D: filters against the complex impedance divider; cut-off frequency at -3 dB;
Clarke-Park with balanced three-phase currents, round trip and power conservation;
second order against `solve_ivp` (overshoot, `ts`, `tr`); margins on cases with a closed
solution; Ziegler-Nichols against the tables. Areas E and F (D43): force on a plane wall against the numerical integral of the pressure on the
surface; buoyancy against the force equilibrium solved with `brentq`; affinity against the
dimensionless coefficients `Q / (n D^3)` and `g H / (n^2 D^2)`; Manning depth fed back into the
formula and critical depth against the numerical minimum of the specific energy; weirs against
the integral of the Torricelli velocity over the head; pressure meters against Bernoulli and
continuity solved with SymPy; isentropic flow with the continuity calculated from `T`, `p`, `V`;
normal shock against the conservation of mass, momentum and energy; critical flow rate
against the numerical maximum of the isentropic flow rate; work of the processes against the numerical
integral of `p dV`; `dS` against the integral of `dQ / T`; cycles against the sum of works and
heats state by state; mixture with a zero heat balance. Strength of materials without tables (D46): thermal stresses against plane Hooke's law solved as a linear system and the compatibility solved with `brentq`; Hertz against the force-approach relation inverted with `brentq` and the integral of the pressure over the contact area; spring stiffness against the torsion energy integrated along the wire (Castigliano); interference pressure against the Lamé radial displacements at the contact solved with `brentq`; principal strains against the eigenvalues of the tensor (numpy) and rosettes against the readings rebuilt from the tensor. Chemistry (D48): weights of the abridged table against the standard table (same interval); formulas against atom counting; balancing against the conservation of each element; pH against the root of the charge balance found with `brentq`; integrated laws against `solve_ivp`; Nernst against `dG = -n F E`; numerical examples from PubChem, OpenStax, LibreTexts and Chem1 in the tests. Nuclear and modern physics (D49): binding energy from atomic masses in u against the B/A published by AME2020; Q of the proton-proton chain against the CODATA masses of proton, alpha and electron (difference equal to the binding of the electrons); Q of alpha decays against the binding energies; threshold against the invariant mass; decays against `solve_ivp`; maximum of the daughter against `minimize_scalar`; Compton against the conservation of energy and momentum; Wien against the numerical maximum of Planck and the band against the integral in wavelength; composed Lorentz transformations against the velocity addition; velocity addition against the rapidities; numerical examples from OpenStax in the tests. Water and steam (D50): IF97 verification tables (5, 15, 33, 35, 36, 42 and the B23 point) to the 9 published digits; inconsistencies at the boundaries 1/3, 2/3 and 2/5 below the maxima of Table 43 and on the saturation line below those of Table 44; cp and v against the finite differences of h and g; cv of region 3 against the derivative of u; inversions (p, h), (p, s), (p, v), (T, v) against the direct state; Rankine against the energy balance and the isentropic transformations. Psychrometrics (D51): 6 states against PsychroLib 2.5.0 (maximum measured deviations: W 2.5e-4 relative, h 1.2e-4, dew point 0.0011 K, wet bulb 0.00077 K, all due to p_ws); state rebuilt from wet bulb, dew point, enthalpy and W; adiabatic saturation balance; mixture, coil and evaporative cooling against the balances of dry air, water and energy. Convection (D52): laminar plate against the numerical integral of the local coefficient; conduction limit of the sphere (Nu = 2); laminar pipe against the exact value 3.65679 (OEIS A282581); Gnielinski at Pr = 1 against the Reynolds analogy; Sieder-Tate viscosity ratio against the example of the source; Grashof and Rayleigh recomputed from the definitions. Heat exchangers, fins and transient (D53): parallel flow and counterflow against solve_ivp and solve_bvp; cross flow against a numerical grid of 100 x 100 and 200 x 200 cells; both mixed against the mean-temperature model; `rating` against the operation `heat_transfer` mode `lmtd`; fins against solve_bvp and the energy balance integrated with quad; transient against solve_ivp. Radiation (D54): 2D view factors against the integral of the definition (kernel cos f1 cos f2 / (2 r) of long strips, dblquad); equal perpendicular plates against plates inclined at 90 degrees; two surfaces against the radiosity system; three surfaces with one re-radiating against the equivalent circuit; shields against the series resistances. Electromagnetism (D55, D56): field of several charges against the numerical gradient of the potential; capacitance of cylinders and spheres against the integrated potential; wire and wall against two wires with the method of images; loop, wire and finite solenoid against the Biot-Savart law integrated with quad; force between wires against B I L; cyclotron against solve_ivp; inductances of toroid and coaxial cable against the integrated flux linkage and magnetic energy; rotating coil against the numerical derivative of the flux. Linkages and gear trains (D57): four-bar positions against the member lengths for 24 angles; classification against the behaviour (full rotation or lock-up); velocities against finite differences; slider-crank against numerical derivatives; numerical examples of the source for the gear train and the differential; planet against the two meshes. Catenary and bearings (D58): length against the arc-length integral; heights and tensions of the supports against equilibrium; small sag against the parabola; Petroff torque against T = f W r and against the integrated shear stress. Trusses (D59): forces and reactions against the method of joints; displacement against virtual work with a unit load; elastic energy against the work of the loads (Clapeyron); global equilibrium of the indeterminate truss; 3D tripod against the equilibrium of the joint. Torsion (D60): ellipse and triangle against the Prandtl stress function written in the test (Laplacian checked by finite differences, J = 2 int phi dA by dblquad, tau = |grad phi| on the boundary); rectangle against finite differences of the Prandtl equation with Richardson extrapolation and against the beta table of the source; thin walls against the shear strain energy; thin closed tube against the exact annulus; circle against `shaft_torsion_bending`.

---

## 13. Start-up

```powershell
uv run oce-math-mcp
```

```powershell
uv run oce-math-mcp --http --host 127.0.0.1 --port 8012
```

```powershell
uv run oce-math-mcp --dump-schemas
```

`--dump-schemas` prints the `name`/`description`/`parameters` list of section 14.
`--dump-schemas --format openai` prints it in the `{"type": "function", "function": {...}}` format
used in the datasets.

`mix_v4` dataset plan (Pillar 1): the examples use these 10 schemas; the tool answers come
from the real server (D12). The `eng_calc` examples wait for phase 2. The TIR datasets (NuminaMath-TIR,
OpenMathReasoning-tir) become calls to `python_sandbox` after measuring how many examples
the interpreter runs without errors.

---

## 14. JSON-Schema contract (identical to `oce_math_mcp_tools.json`)

<!-- SCHEMAS-START -->
```json
[
  {
    "name": "math_eval",
    "description": "Exact arithmetic and scalar functions: Decimal/Fraction/float modes, logs, roots, trig in rad and deg, hyperbolic, combinatorics, statistics and distributions, Italian finance (VAT, withholding, loans, NPV, IRR, DCF), unit conversion, CODATA physical constants via const(\"h\").",
    "parameters": {
      "type": "object",
      "properties": {
        "expr": {"type": "string"},
        "variables": {"type": "object"},
        "exact_mode": {"type": "string", "enum": ["float", "decimal", "fraction"], "default": "decimal"},
        "precision": {"type": "integer", "default": 6, "minimum": 0, "maximum": 30},
        "show_steps": {"type": "boolean", "default": false, "description": "Add numbered steps in Italian to result.steps."}
      },
      "required": ["expr"]
    }
  },
  {
    "name": "math_solve",
    "description": "Symbolic math with SymPy plus SciPy/NumPy: equations, systems, inequalities, derivatives, integrals (single, multiple, line, surface), limits, series, function study, vector calculus, Lagrange, ODE, Laplace, Fourier/FFT, Riccati LQR, linear algebra, sums, recurrences, polynomials, complex numbers. Results are self-verified.",
    "parameters": {
      "type": "object",
      "properties": {
        "operation": {
          "type": "string",
          "enum": [
            "solve", "solve_system", "inequality", "diff", "partial_diff",
            "integrate", "multi_integrate", "line_integral", "surface_integral", "limit", "series",
            "function_study", "simplify", "factor", "expand", "partial_fractions",
            "gradient", "divergence", "curl", "laplacian", "jacobian", "hessian",
            "lagrange_optimize", "ode", "laplace", "inv_laplace", "fourier_fft",
            "riccati_lqr", "matrix", "optimize_num", "complex_ops",
            "summation", "recurrence", "polynomial"
          ]
        },
        "expr": {"type": "string", "description": "Scalar expression, equation with '=', or ';'-separated equations/vector components."},
        "vector_field": {"type": "array", "items": {"type": "string"}, "description": "3D/nD vector field [Fx, Fy, Fz] for divergence/curl/jacobian/line_integral/surface_integral."},
        "constraints": {"type": "array", "items": {"type": "string"}, "description": "Equality/inequality constraints for lagrange_optimize / optimize_num."},
        "variables": {"type": "array", "items": {"type": "string"}},
        "order": {"type": "integer", "default": 1},
        "bounds": {"type": "array", "description": "[a, b] for integrate/summation, [point, dir] for limit, [x0] for series, [[x,a,b],[y,c,d],...] for multi_integrate/surface_integral, [t0, t1] for numeric ode."},
        "ics": {"type": "object", "description": "Initial conditions for ODE e.g. {'y(0)': 1, \"y'(0)\": 0} or recurrence e.g. {'a(0)': 0, 'a(1)': 1}."},
        "matrix_a": {"type": "array"},
        "matrix_b": {"type": "array"},
        "matrix_q": {"type": "array", "description": "State weight Q for riccati_lqr."},
        "matrix_r": {"type": "array", "description": "Control weight R for riccati_lqr."},
        "matrix_op": {
          "type": "string",
          "enum": ["det", "inv", "pinv", "cofactor", "projection", "norm_p", "transpose", "rank", "rref", "lu", "qr", "svd", "cholesky", "eigenvals", "eigenvects", "solve", "multiply"]
        },
        "poly_op": {
          "type": "string",
          "enum": ["roots", "nroots", "factor_list", "discriminant", "resultant", "gcd", "vieta", "coeffs", "degree"],
          "default": "roots",
          "description": "Sub-operation for polynomial."
        },
        "coords": {"type": "string", "enum": ["cartesian", "polar", "cylindrical", "spherical"], "default": "cartesian", "description": "Coordinate system for multi_integrate (Jacobian added automatically)."},
        "time": {"type": "string", "enum": ["continuous", "discrete"], "default": "continuous", "description": "CARE or DARE for riccati_lqr."},
        "domain": {"type": "string", "enum": ["real", "complex", "positive", "integer"], "default": "real"},
        "numeric_samples": {"type": "array", "items": {"type": "number"}, "description": "Signal array for fourier_fft, initial guess for optimize_num."},
        "show_steps": {"type": "boolean", "default": false, "description": "Add numbered steps in Italian to result.steps."}
      },
      "required": ["operation"]
    }
  },
  {
    "name": "number_theory",
    "description": "Integer and modular arithmetic with exact big integers: primality, factorization, divisors, totient, Mobius, primes, gcd/lcm/extended gcd, modular power/inverse/square root/log, primitive roots, Legendre/Jacobi, CRT, Diophantine equations, continued fractions, partitions, digits.",
    "parameters": {
      "type": "object",
      "properties": {
        "operation": {
          "type": "string",
          "enum": [
            "is_prime", "factorint", "divisors", "totient", "mobius",
            "next_prime", "prev_prime", "nth_prime", "prime_pi", "primerange",
            "gcd", "lcm", "gcd_ext", "mod_pow", "mod_inv", "sqrt_mod", "discrete_log",
            "primitive_root", "multiplicative_order", "legendre", "jacobi",
            "crt", "diophantine", "continued_fraction", "partitions", "digits"
          ]
        },
        "n": {"type": ["integer", "string"], "description": "Main integer (decimal string allowed for big numbers)."},
        "m": {"type": ["integer", "string"], "description": "Second integer or modulus."},
        "k": {"type": ["integer", "string"], "description": "Exponent, index or base."},
        "values": {"type": "array", "items": {"type": ["integer", "string"]}, "description": "Integers for gcd/lcm, residues for crt."},
        "moduli": {"type": "array", "items": {"type": ["integer", "string"]}, "description": "Moduli for crt."},
        "expr": {"type": "string", "description": "Equation for diophantine; number for continued_fraction (e.g. 'sqrt(7)', '415/93')."},
        "variables": {"type": "array", "items": {"type": "string"}},
        "limit": {"type": "integer", "default": 1000, "minimum": 1, "maximum": 10000, "description": "Max listed items."},
        "show_steps": {"type": "boolean", "default": false, "description": "Add numbered steps in Italian to result.steps."}
      },
      "required": ["operation"]
    }
  },
  {
    "name": "math_numeric",
    "description": "Numerical math with mpmath, SciPy and NumPy: arbitrary precision evaluation (up to 1000 digits), infinite sums, numerical integrals, roots, constrained and global optimization, linear and integer programming, ODE initial value problems with events (stop when an expression reaches zero), polynomial roots, closed-form identification of constants, curve fitting.",
    "parameters": {
      "type": "object",
      "properties": {
        "operation": {
          "type": "string",
          "enum": ["evalf", "nsum", "quad", "root", "minimize", "linprog", "ode_ivp", "polyroots", "identify", "curve_fit"]
        },
        "expr": {"type": "string", "description": "Expression, equation with '=', ';'-separated equations, or ';'-separated ODE right-hand sides."},
        "variables": {"type": "array", "items": {"type": "string"}},
        "bounds": {"type": "array", "description": "[a, b] or [[x,a,b],...] for quad/nsum, [lo, hi] bracket for root, [[lo, hi], ...] per variable for minimize/linprog, [t0, t1] for ode_ivp."},
        "x0": {"type": "array", "items": {"type": "number"}, "description": "Initial guess, or initial state for ode_ivp."},
        "constraints": {"type": "array", "items": {"type": "string"}, "description": "Constraints like 'x + y <= 4' or 'x*y = 1'."},
        "digits": {"type": "integer", "default": 15, "minimum": 1, "maximum": 1000},
        "method": {
          "type": "string",
          "enum": ["auto", "SLSQP", "trust-constr", "Nelder-Mead", "Powell", "BFGS", "L-BFGS-B", "differential_evolution", "brentq", "hybr", "RK45", "DOP853", "Radau", "BDF", "LSODA"],
          "default": "auto"
        },
        "direction": {"type": "string", "enum": ["minimize", "maximize"], "default": "minimize"},
        "integer": {"type": "boolean", "default": false, "description": "linprog: all variables integer (MILP)."},
        "data_x": {"type": "array", "items": {"type": "number"}},
        "data_y": {"type": "array", "items": {"type": "number"}},
        "samples": {"type": "integer", "default": 11, "minimum": 2, "maximum": 1001, "description": "Output points for ode_ivp."},
        "events": {"type": "array", "maxItems": 10, "items": {"type": "object", "properties": {"expr": {"type": "string"}, "direction": {"type": "string", "enum": ["both", "rising", "falling"], "default": "both"}, "terminal": {"type": "boolean", "default": true}}, "required": ["expr"]}, "description": "ode_ivp only: zero crossings of expressions in the variables. A terminal event stops the integration; a zero exactly at t0 does not count. Example: [{\"expr\": \"y\", \"direction\": \"falling\"}] stops when y comes down to 0."}
      },
      "required": ["operation"]
    }
  },
  {
    "name": "math_verify",
    "description": "Check an answer before giving it: symbolic and random-point equivalence of two expressions, substitution of a solution into equations, derivative and antiderivative checks, numeric comparison with tolerance, counterexample search for inequalities, dimensional check of a formula with units (check_units).",
    "parameters": {
      "type": "object",
      "properties": {
        "operation": {
          "type": "string",
          "enum": ["equivalent", "check_solution", "check_derivative", "check_antiderivative", "check_numeric", "check_inequality", "check_units"]
        },
        "expr": {"type": "string", "description": "Expression, ';'-separated equations, or inequality to check."},
        "expected": {"type": "string", "description": "Claimed result to compare with; for check_units the expected unit of the expression, e.g. \"N\"."},
        "variables": {"type": "array", "items": {"type": "string"}},
        "solution": {"type": "object", "description": "Claimed solution e.g. {'x': '2', 'y': '-1/3'}."},
        "tolerance": {"type": "number", "default": 1e-9},
        "samples": {"type": "integer", "default": 64, "minimum": 1, "maximum": 1000},
        "domain": {"type": "string", "enum": ["real", "complex", "positive", "integer"], "default": "real"},
        "units": {"type": "object", "description": "check_units: unit of each variable, e.g. {\"F\": \"N\", \"rho\": \"kg/m^3\", \"v\": \"m/s\", \"Cd\": \"1\"}; SI prefixes allowed (kN, MPa, mA); degC for Celsius, \"C\" is the coulomb."}
      },
      "required": ["operation", "expr"]
    }
  },
  {
    "name": "python_sandbox",
    "description": "Run a short Python program in a restricted interpreter for problems no other tool covers (counting, brute force, sequences, multi-step computations). Allowed imports: math, cmath, fractions, decimal, itertools, functools, collections, statistics, random, heapq, bisect, sympy, numpy, scipy, mpmath. No files, no network, 5 s limit. Returns printed output and the value of the last expression.",
    "parameters": {
      "type": "object",
      "properties": {
        "code": {"type": "string", "description": "Python source, max 8000 characters. Use print() or end with an expression."}
      },
      "required": ["code"]
    }
  },
  {
    "name": "lean_check",
    "description": "Check a formal proof in Lean 4 with Mathlib. Returns whether it compiles without errors, without sorry and with only the standard axioms, with Lean messages. Code that runs programs (#eval, run_cmd, elab, macro), new axioms and native_decide are rejected. 60 s limit.",
    "parameters": {
      "type": "object",
      "properties": {
        "code": {"type": "string", "description": "Lean 4 source, max 20000 characters. 'import Mathlib' is added if no import is present."},
        "allow_sorry": {"type": "boolean", "default": false}
      },
      "required": ["code"]
    }
  },
  {
    "name": "date_calc",
    "description": "Dates and times in the Italian calendar (Europe/Rome): exact age (also in months and weeks), weekday, differences (days, business days, real elapsed hours across DST) and additions, relative dates, events with DST offsets and reminders, recurring events, time zone conversion, invoice due dates (DFFM), month calendar, Italian national holidays including Easter and Easter Monday.",
    "parameters": {
      "type": "object",
      "properties": {
        "operation": {
          "type": "string",
          "enum": ["now", "age", "weekday", "diff", "add", "resolve_relative", "event_schedule", "invoice_due", "month_calendar", "holidays", "recurring", "convert_tz"]
        },
        "date": {"type": "string"},
        "birth_date": {"type": "string"},
        "ref_date": {"type": "string"},
        "end_date": {"type": "string"},
        "years": {"type": "integer", "default": 0},
        "months": {"type": "integer", "default": 0},
        "weeks": {"type": "integer", "default": 0},
        "days": {"type": "integer", "default": 0},
        "hours": {"type": "integer", "default": 0},
        "minutes": {"type": "integer", "default": 0},
        "business_days": {"type": "boolean", "default": false},
        "target_weekday": {"type": "string", "enum": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]},
        "which": {"type": "string", "enum": ["next", "previous", "this_week", "next_week", "nth_next"], "default": "next"},
        "nth": {"type": "integer", "default": 1},
        "duration_minutes": {"type": "integer", "default": 60},
        "reminders_before": {"type": "array", "items": {"type": "string"}},
        "dffm_days": {"type": "integer"},
        "timezone": {"type": "string", "default": "Europe/Rome"},
        "to_timezone": {"type": "string", "description": "Target IANA zone for convert_tz, e.g. 'America/New_York'."},
        "count": {"type": "integer", "minimum": 1, "maximum": 1000, "description": "Number of occurrences for recurring."},
        "month_week": {"type": "integer", "enum": [1, 2, 3, 4, 5, -1], "description": "recurring monthly by weekday: 1 = first, -1 = last target_weekday of the month."}
      },
      "required": ["operation"]
    }
  },
  {
    "name": "geom3d_calc",
    "description": "3D geometry for CAD, CNC and robotics: vector operations, angles, rotations (Euler, axis-angle), 4x4 homogeneous transforms, quaternions, Denavit-Hartenberg forward kinematics, distances and intersections, cylindrical and spherical coordinates, bolt holes, solid and cross-section properties. For 3D drawing: 2D sketch profiles, circles through 3 points, fillets, triangle solving, curves as point lists (helix, involute, spline, parametric), CAD pick points on faces and edges, patterns, point clouds, mesh and composite part checks (volume, centroid, bounding box).",
    "parameters": {
      "type": "object",
      "properties": {
        "operation": {
          "type": "string",
          "enum": [
            "vec_ops", "angle_between", "rotate_point", "transform_4x4", "quat_euler",
            "dh_kinematics", "point_plane_distance", "line_plane_intersect",
            "point_line_distance", "coord_convert", "pcd_holes",
            "solid_properties", "section_properties",
            "polygon_2d", "circle_3pts", "arc_fillet", "triangle_solve", "curve_points",
            "intersect", "feature_points", "pattern", "point_cloud", "mesh_properties",
            "composite_properties"
          ]
        },
        "v1": {"type": "array", "items": {"type": "number"}},
        "v2": {"type": "array", "items": {"type": "number"}},
        "plane_point": {"type": "array", "items": {"type": "number"}},
        "plane_normal": {"type": "array", "items": {"type": "number"}},
        "angles_deg": {"type": "array", "items": {"type": "number"}},
        "axis": {"type": "array", "items": {"type": "number"}},
        "translation": {"type": "array", "items": {"type": "number"}},
        "quat": {"type": "array", "items": {"type": "number"}},
        "dh_table": {"type": "array", "description": "List of [a, alpha_deg, d, theta_deg] rows for dh_kinematics."},
        "coord_from": {"type": "string", "enum": ["cartesian", "cylindrical", "spherical"]},
        "coord_to": {"type": "string", "enum": ["cartesian", "cylindrical", "spherical"]},
        "params": {"type": "object"},
        "show_steps": {"type": "boolean", "default": false, "description": "Add numbered steps in Italian to result.steps."}
      },
      "required": ["operation"]
    }
  },
  {
    "name": "eng_calc",
    "description": "Engineering design formulas: strength of materials (Hooke, beams, shafts, Von Mises/Tresca, Euler buckling, fatigue, pressure vessels, bolts, welds), machine design (dynamics, vibrations, gears, bearings L10, belts, CNC cutting), fluids and heat (pipes, drag/lift, heat transfer, heatsinks), electrical and control (AC/DC, RLC filters, Clarke/Park, second-order systems, PID, optics, acoustics), material properties; hydrostatics, pumps and hydraulic cylinders, open channels (Manning), flow meters and weirs, compressible flow (isentropic, normal shock, choked), ideal gas states and processes, thermodynamic cycles (Carnot, Otto, Diesel, Brayton), calorimetry; thermal stresses, Hertz contact, helical compression springs, interference (press and shrink) fits, strain transformation and strain-gauge rosettes; CODATA physical constants; chemistry (molar mass from a formula, balancing and stoichiometry, pH of strong and weak acids and bases and buffers, dilution and molarity, Arrhenius and integrated rate laws, Nernst, Faraday electrolysis, Gibbs energy and K); nuclear physics (AME2020/NUBASE2020 nuclide data: masses, binding energy, half-lives, decay modes; radioactive decay law, activity, parent-daughter chains; reaction and decay Q-values with conservation checks and thresholds); modern physics (photon, photoelectric effect, Compton, relativistic de Broglie wavelength, hydrogen-like atoms, blackbody, uncertainty principle); special relativity (gamma, energy and momentum, time dilation, length contraction, velocity addition, Doppler, Lorentz transformation, mass-energy); water and steam with IAPWS-IF97 (state from (p,T), (p,x), (T,x), (p,h), (p,s), (p,v), (T,v); saturation tables; Rankine cycle with pump and turbine efficiencies and reheat); psychrometrics of moist air (state from dry bulb and relative humidity, humidity ratio, wet bulb, dew point or enthalpy; mixing of two streams; heating and cooling coils with condensate; evaporative cooling); convection heat transfer coefficients from Nusselt correlations (laminar flat plate, cylinder in crossflow, sphere, pipe flow laminar and turbulent with Gnielinski, Dittus-Boelter, Sieder-Tate; natural convection on vertical and horizontal plates, horizontal cylinders, spheres and horizontal enclosures); heat exchangers with the effectiveness-NTU method (parallel, counter, crossflow mixed and unmixed, phase change; rating and sizing); straight fins of uniform cross-section with the four tip conditions, fin efficiency and overall efficiency of fin arrays; lumped-capacitance transient with the Biot number check); radiation exchange between gray diffuse surfaces (2D view factors, two-surface enclosures and parallel plates with radiation shields, N-surface enclosures with the radiosity method); electrostatics (Coulomb's law, field, potential and energy of point charges, capacitance of parallel plates, coaxial and eccentric cylinders, parallel wires, wire near a wall, concentric spheres, isolated sphere and disc, capacitor combinations); magnetism (field of a straight wire, loop, finite and long solenoid, toroid; force between parallel wires and on a wire; Lorentz force and cyclotron motion); inductance of solenoids, toroids, coaxial cables, straight wires, loops, rectangles, wire pairs and Wheeler coils; electromagnetic induction (Faraday flux change, rotating coil, moving rod); linkages (Grashof condition and the eight-case classification of four-bar linkages, four-bar position and velocity, slider-crank piston motion, quick-return time ratio) and planetary gear trains (Willis relation, planet speed, differential); catenary cables (from span and length with level or unequal supports, from sag, from horizontal tension) and journal bearings (Sommerfeld number, Petroff friction torque and power loss); pin-jointed 2D and 3D trusses with the direct stiffness method (member forces and stresses, reactions, nodal displacements, strain energy, degree of static indeterminacy); torsion of non-circular sections (circle, annulus, ellipse, equilateral triangle, rectangle by Fourier series, thin-walled open and closed sections by Bredt: torsion constant, maximum shear stress, twist angle); geometric optics (thick lenses, mirrors, refracting surfaces, lens pairs) and wave optics (Malus, Brewster, Fresnel, slits, gratings, thin films, Rayleigh); practical electronics (LED resistor, op-amp circuits, Wheatstone bridge, 555 timer, thermistors); friction in simple machines (inclined plane, leadscrew, rolling resistance); electric machines (transformer, DC, induction and synchronous motors); thin circular and rectangular plates; continuous beams (three moments); local losses at sudden changes of section (Borda-Carnot); stress concentration at holes (Kirsch, Inglis); rotor balancing (one and two planes, ISO grades); rolling without slipping; kinetic theory of gases (molecular speeds, Maxwell-Boltzmann speed distribution, kinetic pressure, mean free path and collision frequency, wall flux and effusion, Graham's law, equipartition and heat capacities); Newtonian gravitation (two bodies, field and potential of point masses, uniform sphere and hollow shell with the binding energy, ring on its axis, effective gravity on a rotating body, Gauss's law, potential difference and the uniform-field approximation).",
    "parameters": {
      "type": "object",
      "properties": {
        "operation": {
          "type": "string",
          "enum": [
            "hooke_elastic", "beam_calc", "shaft_torsion_bending", "stress_criterion",
            "euler_buckling", "fatigue_life", "pressure_vessel", "bolt_joint", "weld_fillet",
            "kinematics_dynamics", "vibration_1dof", "gear_spur", "bearing_life",
            "belt_drive", "cnc_machining", "fluid_pipe", "aero_hydro_force",
            "heat_transfer", "heatsink_thermal", "circuit_ac_dc", "rlc_filter",
            "foc_clarke_park", "control_system", "optics_acoustics", "material_lookup",
            "hydrostatics", "pump_system", "open_channel", "flow_meter",
            "compressible_flow", "ideal_gas", "thermo_cycle", "calorimetry",
            "thermal_stress", "contact_hertz", "spring_helical", "interference_fit",
            "strain_analysis", "physical_constant",
            "molar_mass", "reaction_stoich", "acid_base", "solution_prep",
            "reaction_kinetics", "electrochemistry", "nuclide_data", "radioactive_decay",
            "nuclear_reaction", "modern_physics", "special_relativity", "water_steam",
            "psychrometrics", "convection_correlation", "heat_exchanger_ntu", "fin_heat_transfer",
            "lumped_transient", "radiation_exchange", "electrostatics", "magnetism", "inductance",
            "electromagnetic_induction", "linkage", "planetary_gear", "cable_catenary", "journal_bearing",
            "truss", "torsion_section", "geometric_optics", "wave_optics", "electronics",
            "friction", "electric_machine", "plate_bending", "continuous_beam", "local_loss", "stress_concentration", "rotor_balance", "rolling",
            "kinetic_theory", "gravitation"
          ]
        },
        "params": {
          "type": "object",
          "description": "Operation-specific SI/engineering inputs (all keys explicitly documented & Pydantic-validated)."
        }
      },
      "required": ["operation", "params"]
    }
  }
]
```
<!-- SCHEMAS-END -->
