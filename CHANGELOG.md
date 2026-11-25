# Changelog

Format: one entry per version, newest first. Dates in ISO format (yyyy-mm-dd).

## 0.4.0 - 2026-11-25 (in progress)

New formulas from the coverage measured on 2026-11-25 (233 formulas missing out of 903 in 11 Wikipedia
pages), one group at a time.

- Kinetic theory of gases (D87, spec 4.11.0, `eng_kinetic.py`): operation `eng_calc.kinetic_theory` with 7
  modes: `speeds` (most probable, mean, rms and mean relative speed, mean translational energy, speed of sound
  from the degrees of freedom), `pressure` (p = rho <v^2> / 3, temperature and number density), `distribution`
  (Maxwell-Boltzmann density, cumulative fraction, fraction above a speed without cancellation in the tail,
  fraction between two speeds), `mean_free_path` (from the kinetic diameter, the cross-section or the
  viscosity; collision frequency, time between collisions, Knudsen number), `wall_flux` (impingement rate,
  effusion through a hole, mass flow), `graham` (rate ratio or unknown molar mass) and `equipartition`
  (molar heat capacities, gamma, internal energy). `eng_calc` 79 operations, contract 192. 7 tests in
  `tests/test_kinetic_theory_d87.py` against the moments of the distribution integrated with mpmath, scipy's
  chi distribution, the double integral of the wall flux and page and NIST values. Mutation check: 38
  changed formulas out of 38 caught. The `eng_calc` description in the contract now also names the
  topics of D64-D73 and D87 (it stopped at D60).
- Newtonian gravitation (D88, spec 4.12.0, `eng_grav.py`): operation `eng_calc.gravitation` with 7 modes:
  `two_bodies` (force, potential energy, fields, mu), `point_masses` (field vector and potential at a point of
  up to 1000 point masses, centre of mass, force and energy of a test mass), `sphere` (uniform solid sphere or
  hollow shell inside and outside, mean density, binding energy), `ring` (field and potential on the axis,
  largest field), `rotating_body` (effective gravity at a latitude, plumb-line deflection), `gauss` (flux and
  enclosed mass) and `potential_difference` (without cancellation for close radii; work and the error of the
  uniform-field approximation). `eng_calc` 80 operations, contract 193. 7 tests in
  `tests/test_gravitation_d88.py` against numpy sums, the gradient of the potential, the shell theorem
  integrated from rings with mpmath, Gauss's flux through a cube with scipy, numpy vectors in the rotating frame
  and page values. Mutation check: 39 changed formulas out of 39 caught.
- Review of D87-D88 (D89, spec 4.12.1): `gravitation.rotating_body` gives `g_tangential` and `deflection_deg`
  towards the equator as values >= 0 in both hemispheres (they had the sign of the latitude, while the spec
  said "towards the equator"). The ideal-gas table of the page (section Kinetic theory) is now checked row by
  row in `tests/test_ideal_gas_table_d89.py` (5 tests); the page's adiabatic cell equates the work done by the
  gas with the work done on it (opposite signs): the server follows C_V (T2 - T1) for the work on the gas.
  `CITATION.cff` describes the last released version (0.3.0); a test reads it from the CHANGELOG. Linux tested
  again on the code of 0.4.0 in progress: 1,045 passed, 7 skipped in 58.30 s, self-check
  7.83 s; `RLIMIT_AS` read from `/proc/<pid>/limits` of a worker (1,073,741,824 bytes for memory_mb = 1024). Mutation check: 7 changed
  lines out of 7 caught.
- Audit of the repository (D90-D97, spec 4.13.0): mechanical checks (every operation of the contract answers,
  the keys of `eng_calc` are all in the spec, no broken link), a robustness exploration of D87-D88 and a
  one-pass review by 10 agents (5 scanners by area, 5 verifiers that reproduced each finding); 23 defects
  confirmed and reproduced, all fixed with tests and a mutation check, together with the defects found
  while fixing them (in D90, D91, D93, D96 and D97).
  - `verified` (D90): `diff` gave false for every derivative (the numerical reference went through
    `complex()` inside `mpmath.diff`); now a 40-digit reference, also for `partial_diff`, which compared the
    derivative with itself. `is_zero` took 1e-11 and an undefined `zoo` as zero (`check_solution` accepted
    x = 0 for 1/x = 1); now exact numbers are compared exactly and the numeric fallback is relative.
    Systems of ODEs and `polynomial` nroots, discriminant, resultant, vieta, coeffs and degree said true
    without a check; now each one has an independent check. No solution found means not verified.
  - Counts and bounds (D91): `poisson_pmf`, `binom_pmf` and their CDFs, `gcd`, `lcm`, `mod_inv` and
    `round` refuse non-integer counts (2.5 gave a gamma-based probability, 12.5 was truncated to 12);
    `quad` and `multi_integrate` refuse bounds in the wrong order (they gave 0.0 and "x" for a triangle);
    `multi_integrate` checks its result with a real numerical integral (before, `evalf` left an `Integral`
    and `verified` was false for every multiple integral).
  - `eng_calc` (D92): `control_system` second_order exact rise and settling times for any damping
    (zeta = 1e-5 gave tr_10_90 = 7.30 s instead of 1.0196 s); the beta thermistor never answers below 0 K;
    `bolt_joint` takes the magnitude of the shear and refuses a negative tension; `gear_spur` needs a
    pressure angle between 0 and 45 degrees; `belt_drive` refuses `m_l` without `P`; `psychrometrics`
    refuses `RH2` (dehumidifying coil) when heating.
  - `geom3d_calc` (D93): counts must be integers and are checked before any list is built; a flat mesh has
    `centroid` and `inertia` null with a warning (they were nan); a parametric curve with non-finite points
    or a divergent length is an error, and any other curve whose points or length leave the floating-point
    range is the D74 MATH_ERROR (an involute with base radius 1e306 gave ok = true with length "oo"); two
    messages of `pattern` were still in Italian.
  - `python_sandbox` (D94): numpy integers count for the sequence cap; the shift cap is on the result.
  - `kinetic_theory` and `gravitation` (D95): extreme inputs give the error of D74 instead of the bare
    Python text (119 cases in 11 groups before, 0 after).
  - Parser and pool (D96): exponent-form imaginary literals such as `1e400j` are exact (they failed on
    `Fraction(repr(inf))`); a redundant branch removed; the pool refuses a size below 1.
  - Messages (D97): twelve texts were still in Italian ("Modulo per zero.", "polyroots non converge.",
    "box o cylinder"...). The language test of D81 read only STRING tokens, and on Python 3.12 and later the
    text of an f-string is an FSTRING_MIDDLE token; now it reads both and looks for more Italian words.
- Linux tested (Ubuntu 22.04, x86_64, Python 3.12.13, uv 0.11.20): `--self-check` ok in 8.65 s; suite
  1,026 passed and 7 skipped in 44.91 s. macOS is not tested.

## 0.3.0 - 2026-11-25

Basic sciences: constants, chemistry, nuclear and modern physics, missing mechanical topics.
The whole repository in English, ready for publication, with an installation that others can follow.

- Installation for others (D85): `oce-math-mcp --self-check` starts the server over stdio as an MCP
  client does and runs initialize, tools/list, `math_eval` (2**100 + 1/3, exact) and `eng_calc` (mirror
  f = 0.1 m, d_o = 0.3 m: d_i = 0.15 m); JSON report, exit code 0 only if all 4 pass; Lean reported
  as information only. README Installation rewritten in 5 steps (uv, clone, sync, self-check, client
  registration for Claude Code, Claude Desktop and `mcpServers` JSON files, common problems); new
  `llms-install.md` for AI agents; `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1 by
  reference), `SECURITY.md` (private reporting, protections, scope), GitHub issue forms (bug or wrong
  result, new formula or operation) and pull request template, `CITATION.cff`, project URLs, keywords
  and classifiers in `pyproject.toml`. `tests/test_install_docs.py`: the documents quote only options
  that `--help` lists, the README options table lists all of them, the issue form lists the 10 real
  tools, `CITATION.cff` follows `VERSION`. The stdio test passes `errlog` explicitly:
  `stdio_client` binds `sys.stderr` when `mcp.client.stdio` is imported, and under pytest that stream
  has no file descriptor.
- Lean after a restart (D86): the `TIMEOUT` answer of `lean_check` says that the first check after a
  restart loads Mathlib from disk and to run it again; README note with the measured times.
- Python 3.12 tested (3.12.13 on Windows): 1,019 tests passed, 2 skipped for `pyarrow`, 3 Lean tests
  over the 60 s limit on a cold start and passed when run again (9 of 9).

- Everything in English (D61, spec 4.0.0, breaking): code names, comments, docstrings, tests,
  documents, file names and all server texts (errors, warnings, steps, `formula`, `formatted`).
  Response field `formatted_it` -> `formatted`, step field `text_it` -> `text`; numbers in texts
  as 1,234.56 and dates as ISO. `math_eval`: `vat`, `vat_add`, `vat_remove`, `withholding`
  (arguments net, gross, rate). `date_calc`: key `weekday_it` removed, English holiday names.
  `eng_calc`: English text values (regimes, damping, steam phases, exothermic/endothermic),
  Italian particle aliases removed. Renamed: `misure.py` -> `benchmark.py` (`--benchmark`,
  `--benchmark-md`, `docs/BENCHMARK.md`), `prova_tir.py` -> `tir_replay.py` (`--tir-replay`,
  `--sample`, `--seed`, `--examples`, English report keys), `docs/DECISIONI.md` ->
  `docs/DECISIONS.md`. A structural comparison of every Python file before and after (AST with
  string texts blanked) shows only text, name and separator changes.
- Publication (D61, D63): README with the support buttons and true badges only; tests
  `test_readme_badge.py` (version badge = `VERSION`, no CI badge) and `test_publishable.py`
  (no private addresses, e-mails, tokens, keys, local paths or personal names in the files git
  would publish). Internal notes moved to the project knowledge base; `CLAUDE.md` generic, private
  pointers in the git-ignored `CLAUDE.local.md`. `--sync-spec` mirrors only from
  `OCE_MATH_DATASET_DIR` and `OCE_MATH_RAG_DOC`, no built-in private paths (`test_sync_spec.py`).

- CODATA 2022 physical constants (D47, spec 3.15.0, `constants.py`, `eng_sci.py`): `const("h")` in
  `math_eval` returns the constant as an exact number; `eng_calc.physical_constant` gives value,
  unit, uncertainty, edition and source, or searches the 445 names. 42 short symbols. 14 tests in
  `tests/test_constants_d47.py`: exact SI 2019 constants, exact relations (R = N_A k), Stefan-
  Boltzmann from the base constants, Rydberg within the propagated uncertainty.
- Chemistry (D48, spec 3.16.0, `eng_chem.py`, `data/atomic_weights.json`): 6 `eng_calc`
  operations. `molar_mass` (formula with brackets, complexes and hydrates; IUPAC abridged weights
  with uncertainty), `reaction_stoich` (exact balancing, limiting reagent, amounts), `acid_base`
  (strong and weak acids and bases, buffers; exact charge balance), `solution_prep`
  (dilution, molarity), `reaction_kinetics` (Arrhenius, integrated laws of order 0, 1, 2),
  `electrochemistry` (Nernst, Faraday, Gibbs energy and K). New units: concentration
  (mol/L), molar energy, charge (C, Ah, mAh), mmol. 46 tests in `tests/test_chem_d48.py`,
  with the numerical examples from PubChem, OpenStax, LibreTexts and Chem1.
- Nuclear and modern physics (D49, spec 3.17.0, `eng_nuc.py`, `eng_modern.py`,
  `data/nuclides.json.gz`): 5 `eng_calc` operations. `nuclide_data` (AME2020 and NUBASE2020:
  masses, binding energy, half-life, decays, abundances of 5,383 nuclides),
  `radioactive_decay` (law, activity, parent-daughter chains with Bateman), `nuclear_reaction`
  (Q with nucleon, charge and lepton number checks, threshold, product energies),
  `modern_physics` (photon, photoelectric effect, Compton, relativistic de Broglie, hydrogen-like
  atoms, black body, uncertainty), `special_relativity` (kinematics, energy and momentum,
  time dilation, length contraction, velocity addition, Doppler, Lorentz, mass-energy).
  New units: keV, MeV, GeV, pm, fm, angstrom, u, Da, MeV/c2, Bq and Ci, years, MeV/c, MHz-PHz,
  km/s, MV, decay constant. 36 tests in `tests/test_nuclear_d49.py`. Data attribution
  in `NOTICE`.
- Water and steam (D50, spec 3.18.0, `eng_steam.py`, `if97_data.py`): operation
  `eng_calc.water_steam` with IAPWS-IF97 (IAPWS R7-97(2012)). `state` from (p, T), (p, x), (T, x),
  (p, h), (p, s), (p, v), (T, v): region, phase, v, rho, h, u, s, cp, cv, speed of sound,
  quality; `saturation` from p or T (saturated steam table); `rankine` with pump and turbine
  efficiencies, reheat, power or flow rate. Coefficients generated from the official PDF.
  New units: specific volume (m3/kg, L/kg, dm3/kg, cm3/g), mass flow rate (kg/s, kg/h,
  t/h, kg/min). 31 tests in `tests/test_steam_d50.py`: verification tables of the document to 9
  digits, inconsistencies at the boundaries (tables 43 and 44), thermodynamic identities with finite
  differences, inversions, cycle balance.
- Date of D49 corrected: 2026-11-24, not 2026-11-25 (DECISIONS and spec).
- Psychrometrics (D51, spec 3.19.0, `eng_psychro.py`): operation `eng_calc.psychrometrics` with the
  ASHRAE 2017 ch. 1 ideal-gas model and p_ws from IAPWS-IF97. `state` from dry bulb and one
  of relative humidity, W, wet bulb, dew point, enthalpy; `mixing` of two streams; `heating_cooling`
  (heating or cooling coil, with condensation); `evaporative_cooling`. Pressure or altitude (ISO 2533).
  14 tests in `tests/test_psychro_d51.py`, with the comparison against PsychroLib 2.5.0 in 6 states.
- Convection (D52, spec 3.20.0, `eng_conv.py`): operation `eng_calc.convection_correlation`, 9
  geometries with the Nusselt correlations (laminar plate, Churchill-Bernstein, Faeth, laminar
  tube, Gnielinski, Dittus-Boelter, Sieder-Tate, Churchill-Chu, Incropera, Yuge,
  Globe-Dropkin); warnings outside the validity range; Q with area and temperature difference.
  10 tests in `tests/test_convection_d52.py`.
- Heat exchangers, fins and transients (D53, spec 3.21.0, `eng_hx.py`): operations
  `eng_calc.heat_exchanger_ntu` (e-NTU method: effectiveness, inverse NTU, `rating`, `sizing`;
  parallel flow, counterflow, cross flow with and without mixing, phase change),
  `eng_calc.fin_heat_transfer` (constant cross-section fins, 4 tip conditions, efficiency,
  arrays) and `eng_calc.lumped_transient` (lumped parameters with the Biot number). New quantity
  type `thermal_conductance` (W/K, kW/K). 16 tests in `tests/test_heat_exchanger_d53.py`.
- Radiation between surfaces (D54, spec 3.22.0, `eng_rad.py`): operation
  `eng_calc.radiation_exchange` with the modes `view_factor_2d` (4 2D geometries), `two_surface` (cavity
  of two grey surfaces, enclosed surface, parallel plates with shields) and `enclosure` (N surfaces,
  radiosity method, with sum and reciprocity checks). 8 tests in `tests/test_radiation_d54.py`.
- Electrostatics (D55, spec 3.23.0, `eng_em.py`): operation `eng_calc.electrostatics` with the modes
  `coulomb`, `point_charges`, `capacitor` (8 geometries) and `combination`; charges also in mC, uC, nC, pC.
  6 tests in `tests/test_electrostatics_d55.py`.
- Magnetism, inductance and induction (D56, spec 3.24.0, `eng_mag.py`): operations `eng_calc.magnetism`
  (7 modes), `eng_calc.inductance` (8 geometries) and `eng_calc.electromagnetic_induction` (3 modes); new
  quantity types `magnetic_flux_density` (T, mT, uT, nT, G) and `magnetic_flux` (Wb, mWb, uWb).
  8 tests in `tests/test_magnetism_d56.py`.
- Mechanisms and gear trains (D57, spec 3.25.0, `eng_linkage.py`): operations `eng_calc.linkage` (Grashof,
  four-bar linkage, slider-crank, time ratio) and `eng_calc.planetary_gear` (epicyclic gear train
  with the planet gear, differential). 7 tests in `tests/test_mechanisms_d57.py`.
- Catenary and bearings (D58, spec 3.26.0, `eng_cable.py`): operations `eng_calc.cable_catenary`
  (3 modes, supports also at different heights) and `eng_calc.journal_bearing` (Sommerfeld, Petroff). 4 tests in
  `tests/test_catenary_bearing_d58.py`.
- Trusses (D59, spec 3.27.0, `eng_truss.py`): operation `eng_calc.truss` (stiffness method
  in 2D and 3D, reactions, displacements, energy, degree of static indeterminacy). 6 tests in
  `tests/test_truss_d59.py`.
- Torsion of non-circular sections (D60, spec 3.28.0, `eng_torsion.py`): operation
  `eng_calc.torsion_section` (7 shapes, torsion constant, maximum shear stress, twist). 7 tests in
  `tests/test_torsion_d60.py`.
- Geometric and wave optics (D64, spec 4.1.0, `eng_optics.py`): operations `eng_calc.geometric_optics`
  (thick lens with principal planes, also in a medium; mirror; refracting surface; two lenses with the
  afocal case) and `eng_calc.wave_optics` (Malus, Brewster, Fresnel, double and single slit, grating with
  resolving power, thin film, Rayleigh). `eng_calc` 69 operations, contract 182. 17 tests in
  `tests/test_optics_d64.py` against exact ray tracing, ABCD matrices, boundary conditions, phasor sums
  and the characteristic matrix of a film.
- Practical electronics (D65, spec 4.2.0, `eng_electronics.py`): operation `eng_calc.electronics` (LED series
  resistor with a switch drop, op amps: inverting, non-inverting with finite open-loop gain and bandwidth,
  summing, difference, integrator, differentiator; Wheatstone bridge balance, output and galvanometer current;
  555 astable and monostable; thermistor B and Steinhart-Hart models both ways and fits from 2 or 3 points).
  `eng_calc` 70 operations, contract 183. 11 tests in `tests/test_electronics_d65.py` against Kirchhoff laws with
  a finite-gain op amp, a single-pole model, nodal analysis of the bridge, the capacitor ODE of the 555 and the
  examples of the source pages; 5 mutations of the formulas out of 5 caught.
- Friction in simple machines (D66, spec 4.3.0, `eng_friction.py`): operation `eng_calc.friction` (inclined plane
  with a parallel or horizontal push, leadscrew with square or trapezoidal thread and collar, rolling resistance).
  4 tests in `tests/test_friction_d66.py` against the equilibrium of the block, the 3D force balance on the thread
  flank (gap of the page formula up to 2.5e-4) and the moment balance of the wheel.
- Electric machines (D67, spec 4.4.0, `eng_machines.py`): operation `eng_calc.electric_machine` (ideal transformer
  and EMF equation, DC motor, induction motor with the per-phase circuit and breakdown torque, synchronous machine).
  `eng_calc` 72 operations, contract 185. 5 tests in `tests/test_machines_d67.py` against the motor ODEs, coupled
  coils, the flux derivative and phasor circuits. Mutation check: 11 changed formulas out of 11 caught (D66 and D67).
- Thin plates (D68, spec 4.5.0, `eng_plates.py`): operation `eng_calc.plate_bending` (circular plate clamped or
  simply supported, rectangular simply supported plate with the Navier series; deflection, moments, stresses).
  2 tests in `tests/test_plates_d68.py` against finite differences of the biharmonic equation with Richardson
  extrapolation and the general axisymmetric solution.
- Continuous beams (D69, spec 4.6.0, `eng_beams.py`): operation `eng_calc.continuous_beam` (theorem of three
  moments with uniform and point loads, different EI, settlements, fixed ends; reactions and span moments).
  `eng_calc` 74 operations, contract 187. 2 tests in `tests/test_continuous_beam_d69.py` against the stiffness
  method in 6 cases. The settlement sign printed on the source page is wrong for its own conventions (D69).
  Mutation check: 10 changed formulas out of 10 caught (D68 and D69).
- Local losses (D70, spec 4.7.0, `eng_losses.py`): operation `eng_calc.local_loss` (Borda-Carnot equation;
  sudden expansion with loss coefficient, sudden contraction with the Weisbach contraction coefficient or a given
  one; head loss, pressure change, loss coefficients referred to both velocities). `eng_calc` 75 operations,
  contract 188. 3 tests in `tests/test_local_loss_d70.py` against the momentum balance of the control volume.
  Mutation check: 8 changed formulas out of 8 caught.
- Stress concentration (D71, spec 4.8.0, `eng_holes.py`): operation `eng_calc.stress_concentration` (Kirsch
  stress field around a circular hole with uniaxial or biaxial far field, edge extremes and K_t; Inglis
  elliptical hole). `eng_calc` 76 operations, contract 189. 3 tests in `tests/test_stress_concentration_d71.py`:
  equilibrium and compatibility by finite differences, traction-free edge, far field, scan of the edge.
  Mutation check: 10 changed formulas out of 10 caught.
- Rotor balancing (D72, spec 4.9.0, `eng_balance.py`): operation `eng_calc.rotor_balance` (static and couple
  unbalance of point masses, centrifugal force and couple, one-plane and two-plane correction, ISO balance
  grade). `eng_calc` 77 operations, contract 190. 3 tests in `tests/test_rotor_balance_d72.py` against the 3D
  vector sum of the centrifugal forces and moments. Mutation check: 12 changed formulas out of 12 caught.
- Rolling without slipping (D73, spec 4.10.0, `eng_rolling.py`): operation `eng_calc.rolling` (incline or force
  through the centre; acceleration, friction needed, minimum static friction, kinetic energy). `eng_calc` 78
  operations, contract 191. 2 tests in `tests/test_rolling_d73.py` against Newton's equations solved as a
  linear system and the energy route. Mutation check: 11 changed formulas out of 11 caught.
- Finite results (D74, spec 4.10.1): a random exploration of the `eng_calc` operations with extreme inputs
  (1e300, 1e-300) found ok answers with "oo" or "nan" (16 cases in D64-D73, 3 in older operations).
  `eng_calc` now turns a non-finite result into `MATH_ERROR`. 20 tests in `tests/test_finite_results.py`;
  mutation check of the guard 4 out of 4 caught.
- Numeric edge cases (D75), found by the random exploration with seeds taken from the test suite: `torsion_section`
  with a zero torque failed with 0/0 (the section modulus does not depend on the torque); `modern_physics`
  blackbody far in the Wien tail overflowed (now e^-x through the logarithm, exact to 1e-12 against mpmath);
  `control_system` second_order heavily overdamped lost digits by cancellation in the slow pole (zeta = 1e4: settling
  time wrong from the 9th digit; zeta = 1e9: division by zero), now the stable form wn / (zeta + sqrt(zeta^2 - 1)).
  3 tests in `tests/test_numeric_edges_d75.py` (references with mpmath at 50 digits); mutation check 10 out of 10.
- Cancellation (D76): four formulas lost digits by subtracting close numbers. Measured against mpmath: Compton
  shift at 0.01 degrees 3.5e-9, view factor of parallel strips with w / L = 1e-6 8.9e-5, perpendicular strips with
  w_j / w_i = 1e-6 1.3e-10, inclined strips at 179.999 degrees 8.0e-7. Now 2 sin^2(theta / 2), rationalised
  differences of roots with `math.hypot` (no overflow: a huge perpendicular strip gives the limit 1/2, before 0 or
  nan) and 180 - alpha computed in degrees: all within 1.5e-15. 2 tests in `tests/test_cancellation_d76.py`;
  mutation check 7 out of 7.
- Cancellation, second group (D77): a search for an exponential or a logarithm followed by a subtraction found
  seven more places. Measured against mpmath: RC and RL charge at t / tau = 1e-8 1.0e-8; RLC step response at
  small times up to 2.2e-8, heavily overdamped circuit 8.6e-9 (root s1 = -alpha + sqrt(alpha^2 - omega0^2)); lumped
  heat Q at small t 1.1e-6, time from a temperature close to T_0 3.6e-8; decayed fraction at small lambda t
  3.8e-8; Bateman chain with half-lives equal within 1e-11 1.6e-5 (t_max 2.5e-6); integrated kinetics of order 1
  and 2 with c = c0 (1 - 1e-12) 1.1e-4; fin temperature ratio 1.6e-11. Now expm1 and log1p, s1 = omega0^2 / s2,
  the Taylor series of the RLC response for (alpha + omega0) t <= 0.5, 2 e^(-alpha t) sinh(s t) near critical
  damping and an exact `d == 0` test for equal half-lives: all within 2.3e-15. 7 tests in
  `tests/test_cancellation_d77.py`; mutation check 18 out of 18.
- Cancellation, third group (D78): kinetic energies and small relative changes. Measured against mpmath:
  relativistic kinetic energy (gamma - 1) m c^2 at beta = 1e-6 8.9e-5, from a small momentum 9.5e-6, de Broglie
  kinetic energy of an electron at 1000 m/s 1.4e-5, a kinetic energy of 1e-19 J through gamma and back 7.4e-11;
  Compton electron energy E - E' at 0.1 degrees 3.3e-9; Doppler z = r - 1 at beta = 1e-9 8.2e-8 and beta from
  f / f_observed = 1 + 1e-9 8.1e-8; damped frequency sqrt(1 - zeta^2) at zeta = 1 - 1e-8 2.8e-10. Now gamma - 1 is
  kept apart ((gamma beta)^2 / (gamma + 1), K / (m c^2), q^2 / (sqrt(1 + q^2) + 1)), E - E' = E shift / (lambda +
  shift), z from a difference of the given values, z = 2 beta / ((1 - beta)(r + 1)) and sqrt((1 - zeta)(1 + zeta)):
  all within 3.6e-15 (Mp within 1.6e-14, where its exponent of about 222 multiplies the rounding). 5 tests in
  `tests/test_cancellation_d78.py`; mutation check 18 out of 18.
- Logarithm of a ratio close to 1 (D79): `log_ratio(a, b)` in `eng_base` gives ln(a / b) as log1p((a - b) / b) when
  b / 2 <= a <= 2 b, where a - b is exact; ln(a / b) loses the digits of the rounding of a / b. Measured against
  mpmath: LMTD with d1 / d2 = 1 + 1e-11 1.1e-5 (a balanced counterflow exchanger), coaxial capacitor with a gap of
  1e-9 8.7e-8, Arrhenius Ea from temperatures 1e-9 apart 4.8e-8, cylindrical wall, toroid, coaxial inductance and
  isothermal work with a ratio 1 + 1e-6 2.8e-11. Also 1/T1 - 1/T2 = (T2 - T1) / (T1 T2) in the two-point fits
  (Arrhenius, thermistor B) and in the capacitance of concentric spheres; the LMTD branch for equal differences
  only for d1 == d2 (before: within 1e-12, an approximation). After: all within 5e-15. 5 tests in
  `tests/test_cancellation_d79.py`; mutation check 15 out of 15.
- Differences of powers and of reciprocals (D80): `sq_diff(a, b)` = (a - b)(a + b) and `pow4_diff(a, b)` =
  (a - b)(a + b)(a^2 + b^2) in `eng_base`. Measured against mpmath: radiation with T1 = T2 (1 + 1e-9) 1.0e-8 (also
  between parallel plates), eccentric cylinders almost touching 1.5e-9 (acosh close to 1, now log1p of an exact
  difference), mirror with the object 1e-8 f beyond the focus 4.5e-9 (1/f - 1/d_o, now (d_o - f) / (f d_o)), thick
  cylinder with t = 1e-6 r_i 4.4e-11, hollow shaft and annulus with d_i = d (1 - 1e-6) 2.1e-11; the interference fit
  uses the same differences. After: all within 4.1e-15. 5 tests in `tests/test_cancellation_d80.py`; mutation
  check 16 out of 16.
- Italian left in five error messages (D81): "Serve ..." and "Per 'multiply' serve matrix_b." in `date_calc`,
  `eng_solid`, `geom3d_calc` and `math_solve`, now in English. The vocabulary check of D61 could not see them,
  because "serve" and "per" are English words too; `tests/test_english_messages.py` reads every string literal
  of the package and fails on such Italian verbs and phrases.
- Lenses, refracting surface, close wires and taut cables (D82). Measured against mpmath, case 1e-8: thin lens
  near the focus and mirror or lens with d_i close to -d_o 4.5e-9 (now f d_o / (d_o - f) and d_o d_i / (d_o +
  d_i)); thin meniscus with R2 = R1 (1 + 1e-8) 4.5e-9 (now (R2 - R1) / (R1 R2)); refracting surface with the
  object near the focus 5.2e-9 (the numerator (n2 - n1) d_o - n1 R now in exact rationals); two wires, wire and
  wall, two-wire line almost touching 1.0e-9 (new `acosh1p(y)` = acosh(1 + y) in `eng_base`); catenary from the
  tension with a = 1e4 H 7.1e-8 in the sag (now 2 a sinh^2(x/(2a))), taut cable with L = H (1 + 1e-8) 4.8e-9 in
  a (the length equation now H (sinh(u)/u - 1) = sqrt(L^2 - v^2) - H, with a series for small u and L^2 - v^2 -
  H^2 in exact rationals), sag of 1e-6 m on a span of 100 m 9.6e-2 in a (acosh of 1 + h/a). After: all within
  4.1e-15. The thin lens now answers d_i = f and d_i = -d_o with a clear message instead of "ZeroDivisionError:
  float division by zero". 9 tests in `tests/test_cancellation_d82.py`; mutation check 20 out of 20.
- Hertz contact in a concave seat close to conforming (D83): 1/R = 1/R1 + 1/R2 lost digits with R2 = -R1 (1 +
  1e-8), 2.7e-9 in R (5.0e-5 with 1e-12); now R = R1 R2 / (R1 + R2), within 1e-13. A last sweep found nothing
  else: the exploration with two parameters tied (equal, opposite, double, half) over 211 modes gave only the
  absurd inputs already accepted in D74 and D75. 1 test in `tests/test_cancellation_d83.py`; mutation check 2
  out of 2.
- Extreme ranges after D79-D83 (D84), found by an independent review: products of two lengths (f d_o, R1 R2,
  n2 R d_o, L^2) overflowed or underflowed where the old reciprocals did not. A mirror with d_o = 1e308 (infinity
  written as a number) answered "at the focus", a thin lens with f = 1e-200 answered d_i = 0, a meniscus with
  radii of 1e160 was "afocal"; now a ratio first and one product, or an exact rational quotient. The thin lens
  refuses f = 0 and d_i = 0. Lame's sigma_r at the walls is now the pressure itself (it lost up to 3e-11 for a
  thin wall), and the thick sphere uses r_o^3 - r_i^3 = (r_o - r_i)(r_o^2 + r_o r_i + r_i^2) (8.6e-12 before).
  `fmt_float` raised decimal.InvalidOperation from 1e22 on (28 digits of the Decimal context), so ten operations
  answered "InvalidOperation" for large finite results (bearing life with P = 1e-9 N, for one); now scientific
  notation there, the same text below. 9 tests in `tests/test_extreme_ranges_d84.py`; mutation check 20 out of 20.

## 0.2.0 - 2026-11-24

`eng_calc` areas A, B, C and D (D39): the formula lives in the server. Spec 3.7.0, JSON schemas unchanged.

- 6 operations ready out of 25: `hooke_elastic`, `beam_calc` (simply supported, cantilever, fixed-fixed;
  point force or uniform load), `shaft_torsion_bending` (also from power and rpm, minimum diameter
  with Von Mises or Tresca), `stress_criterion` (principal stresses, Von Mises, Tresca),
  `euler_buckling` (4 constraints, slenderness, Johnson), `pressure_vessel` (thin wall and Lamé,
  cylinder and sphere).
- Area B, 6 operations: `kinematics_dynamics` (modes `projectile` without air, `uniform_acceleration`,
  `energy`, `rotation`, `steiner`, `circular`), `vibration_1dof` (free and forced), `bearing_life`
  (L10, hours, required C), `belt_drive` (exact length, Euler-Eytelwein, V-groove, centrifugal),
  `cnc_machining` (turning, milling, drilling), `gear_spur` (spur and helical teeth,
  forces, contact ratio, Lewis).
- Area C, 4 operations: `fluid_pipe` (Reynolds, Colebrook-White, Darcy-Weisbach, local
  losses, pump head and power, Bernoulli), `aero_hydro_force` (drag,
  lift, Mach, terminal velocity), `heat_transfer` (layered wall, cylinder, convection,
  radiation, LMTD), `heatsink_thermal` (thermal chain and required R_sa).
- Area D, 5 operations: `circuit_ac_dc` (Ohm, resistors, divider, single-phase and three-phase, voltage
  drop in cables with IEC 60028 copper), `rlc_filter` (RC, RL, RLC), `foc_clarke_park`
  (direct and inverse, amplitude or power scale), `control_system` (second order, gain and phase
  margins, Ziegler-Nichols), `optics_acoustics` (Snell, thin lens, decibel, sound
  pressure level, sum of levels, distance, wavelength).
- The other 4 (`fatigue_life`, `bolt_joint`, `weld_fillet`, `material_lookup`) need data tables
  with sources: they answer `NOT_IMPLEMENTED` with the list of ready ones.
- Code split by area: `eng_base.py` (reading values with units), `eng_struct.py` (area A),
  `eng_mech.py` (area B), `eng_thermo.py` (area C), `eng_elec.py` (area D); `eng_calc.py` loads them.
  New unit categories: density, viscosity, conductivity, heat transfer, thermal
  resistance, electrical quantities; temperatures in C, K and F.
- Values in `params` as an SI number or `[value, "unit"]`; a unit of the wrong type and an
  unexpected key give `INVALID_ARGUMENT` with the reason. Results in SI with `units`,
  formulas in `formula_used`, English text in MPa and mm.
- Every formula tested with an independent method (beams against SymPy `Beam` on 8 cases, Lamé
  with equilibrium, Von Mises from eigenvalues; area B against numerical integration of the motion and the
  geometric construction of belt and gears; area C against Hagen-Poiseuille, Haaland,
  Swamee-Jain and the integration of the exchanger; area D against complex impedances and
  `solve_ivp`): 35 tests in `tests/test_eng_calc.py`, 16 in `tests/test_eng_calc_b.py`, 15 in
  `tests/test_eng_calc_c.py`, 23 in `tests/test_eng_calc_d.py`.
- `math_eval.unit_factor`: SI factor of a `unit_convert` unit, used by `eng_calc`.
- `math_numeric.ode_ivp`: new field `events` (D41, spec 3.8.0): zeros of expressions in the
  variables, terminal or not, with direction; a zero exactly at t0 does not count (sign from the
  derivatives along the motion). Result with `t_end`, `t_events`, `y_events`, `stopped_by`. Without
  `events` nothing changes. 15 tests in `tests/test_ode_events.py` (closed form of the projectile,
  exact linear drag, zeros of the oscillator).
- `math_verify.check_units` (D42, spec 3.9.0) and field `units`: tells whether the dimensions of a
  formula are consistent (sums, functions, exponents, two sides), with the SI factor and a warning for
  units of the same dimension but different scale (km/h, degrees). Units from `sympy.physics.units`
  by name, SI prefixes and aliases; hand-written unit parser (`unitdim.py`). The unit table
  is built at start-up of the calculation process (1.46 s measured), not inside the
  timeout: first call 0.002 s. 43 tests in `tests/test_check_units.py`.
- `python_sandbox` (D40, approved by the maintainer, spec 3.10.0): `vars` as a variable name or argument
  (the builtin stays forbidden), modules `time` (clocks only, no `sleep`) and `datetime` with a
  closed list of attributes. Cases taken from the OpenMathReasoning measurement: 16 tests in
  `tests/test_sandbox_d40.py`.
- `eng_calc` areas E and F (D43, spec 3.11.0, extended contract): 8 new operations from the physics
  catalogue. Fluids (`eng_hydro.py`): `hydrostatics` (pressure, Archimedes, force on a
  flat wall), `pump_system` (power, affinity laws, NPSH, hydraulic cylinder), `open_channel`
  (Manning, normal and critical depth, partially full pipe), `flow_meter` (orifice, nozzle,
  Venturi, weirs), `compressible_flow` (isentropic, A/A*, normal shock, critical flow rate).
  Thermodynamics (`eng_gas.py`): `ideal_gas` (state and 5 processes with W, Q, dU, dH, dS),
  `thermo_cycle` (Carnot, Otto, Diesel, Brayton also with efficiencies), `calorimetry` (sensible,
  latent, mixture). New units: specific heat, energy per kg, molar mass, moles. 29 tests in
  `tests/test_eng_calc_ef.py`, each against an independent method (numerical integrals,
  conservation, balances, CODATA molar volume).
- `eng_calc` complete, 33 operations out of 33 (D44, spec 3.12.0, `eng_tables.py`): `fatigue_life`
  (Marin, Goodman, Gerber, Soderberg, Basquin, EN 1993-1-9 curves, Miner), `bolt_joint` (ISO 262,
  ISO 898-1, EN 1993-1-8 tables 3.1 and 3.4, torque K and VDI 2230, friction), `weld_fillet` (EN
  1993-1-8 4.5.3, simplified and directional method), `material_lookup` (13 materials, source for
  each property, `null` if not documented). Sources collected from the web and the offline library,
  conflicts resolved in D44. 17 tests in `tests/test_eng_calc_tab.py`, with the numerical examples
  of the sources. The contract test now requires that every enum entry has code.
- 10 new modes in existing operations (D45, spec 3.13.0, JSON schemas unchanged, `eng_dyn.py`):
  `kinematics_dynamics` with `collision`, `orbit` (Kepler, vis-viva, Hohmann), `gyroscope`,
  `flywheel`; `optics_acoustics` with `doppler`, `string`, `pipe`, `intensity`; `circuit_ac_dc`
  with `impedance` (R, L, C in series or in parallel) and `transient` (RC, RL, RLC step response).
  Units `ms` and `us`. 19 tests in `tests/test_eng_calc_d45.py` against independent methods
  (integrated two-body problem, wave fronts, string eigenvalues, integrated circuits).
- 5 new `eng_calc` operations, strength of materials without tables (D46, spec 3.14.0, `eng_solid.py`):
  `thermal_stress` (free expansion, constrained bar or with gap, biaxial plate),
  `contact_hertz` (spheres and cylinders, maximum pressure, maximum shear stress below the
  surface from the Johnson stresses), `spring_helical` (stiffness, Wahl and Bergsträsser,
  ends, solid length, natural frequency), `interference_fit` (Lamé, different materials,
  hollow shaft, torque and axial force, fitting temperature), `strain_analysis`
  (transformation, principal strains, rectangular, delta or arbitrary-angle rosettes,
  principal stresses). Thermal expansion units (`1/K`, `um/(m*K)`, `ppm/K`). 34 tests in
  `tests/test_eng_calc_d46.py` against independent methods (plane Hooke as a linear system,
  inverted force-approach with `brentq`, integral of the pressure, torsion energy of the
  wire, Lamé displacements at the contact, tensor eigenvalues, discrete chain for the natural frequency
  of the spring) and 9 with the numerical examples from the sources (Shigley, Univ. of Washington, ANSYS VM27,
  Roylance, TN-515). 20 sources verified in the project knowledge base.
- `--tir-replay`: recognises the truncation of OpenMathReasoning outputs (`<output cut>`, also in
  the middle of a line) and compares the beginning also without spaces (the notebook wraps lines inside long
  dictionaries and matrices).

## 0.1.2 - 2026-11-24

D13 measurement on public TIR datasets. Only the test program changes: tools and schemas unchanged.

- `--tir-replay` reads NuminaMath-TIR (python blocks followed by output) and OpenMathReasoning
  (`<tool_call>` followed by output), `.jsonl` and `.parquet` files (`pyarrow` optional).
- `--sample N` and `--seed S`: repeatable random sample, with a fingerprint of the indices.
- `--examples N`: examples saved for each error type, with the dataset line number.
- The test uses the server process pool: real 5 s timeout and a new process after every
  call (D29); `--no-pool` runs it in the same process.
- Comparison: previous blocks placed first when a block gives `NameError`; expected error outputs
  (traceback, `Timed out after`) counted separately; truncated outputs (`... (output truncated)`)
  compared by their beginning; environment differences (numpy 1 and 2, notebook `Fraction`, tuples over
  several lines, IPython warning) corrected and counted in `equal_after_normalisation`.
- Errors counted by type (`errors_by_kind`).
- Previous blocks run as in a notebook: each inside `try`, so a block that was already failing
  in the dataset does not stop the ones after it. Jupyter tracebacks and `Timed out` (without
  "after") recognised as expected error outputs.
- Sample read in streaming mode: of the `.parquet` files only the row groups that contain the chosen
  indices are read (OpenMathReasoning tir: 72 files, 1,718,466 rows).
- Number comparison: integers compared exactly (no `OverflowError` on huge integers);
  a comparison that fails counts as "different", it does not stop the measurement.
- D37 confirmed by the maintainer: Lean and Mathlib on a local disk.
- D39 (maintainer's choices on physics): all 25 `eng_calc` operations, check of the essential
  formula, FEM on standby. Catalogue of the gaps in the project knowledge base.

## 0.1.1 - 2026-11-24

Sandbox ready for the dataset TIR programs (D13, D38). Spec 3.4.0, JSON schemas unchanged.

- `--tir-replay FILE...`: runs in `python_sandbox` the calculation calls of the TIR JSONL files and
  compares the result with the recorded answer. On the 2 Italian files of the dataset: 120 out of 120
  equal, 0 errors.
- `--benchmark-md FILE`: writes the table of measurements in Markdown (`docs/BENCHMARK.md`).
- D38: `python_sandbox` accepts `from ... import *` (only names that pass the checks), `global`
  inside a function, `sympy.S` with constants and numbers, `nsimplify` of numbers. `class` stays
  forbidden. Probe of 30 TIR constructs: 29 pass (`tests/test_tir_constructs.py`, outputs equal to
  real Python).
- Fixed: comparisons in the sandbox always returned true/false, so `x**2 > 4` did not
  reach `solveset` as a SymPy relation.
- The `sympify` protection reads the numeric forms of `mpmath.identify` with the AST parser
  (`"(sqrt(2)/2)"`), never with `eval`.
- Test D31 extended to the sandbox (star import and `nsimplify`): `eval` and `exec` are used only by SymPy and
  by `namedtuple`/`dataclasses` of the standard library when loading modules, never on the model
  text.
- The real stdio test (`tests/test_lean_server.py`) also runs a TIR program with star import,
  `S.Reals`, comparison and `nsimplify`. HTTP test on port 8012: 10 tools, same result.
- `docs/BENCHMARK.md` regenerated on 0.1.1 with real Lean.
- `.gitignore` excludes `.hypothesis/`: the Hypothesis cache writes full local paths.

## 0.1.0 - 2026-11-24

First working version: 9 tools out of 10 (112 operations out of 137; `eng_calc` in phase 2).

- Code of the tools `math_eval`, `math_solve`, `number_theory`, `math_numeric`, `math_verify`,
  `python_sandbox`, `lean_check`, `date_calc`, `geom3d_calc`; MCP server over stdio and HTTP.
- D27: the `sympify` protection accepts the fraction of two integers (SymPy uses it internally).
- D28, D29: tighter sandbox; calculation process replaced after every `python_sandbox`.
- D30: holidays from 2001 with a warning for earlier years; Easter from 1583.
- D31: test that intercepts `eval` and `exec` during calculations (only SymPy uses them, never on the text
  of the model).
- D32: calculation processes as subprocesses with dedicated pipes. First call over stdio from
  132.15 s to 10.80 s.
- D33: `ruff check`, `ruff format`, `basedpyright` green; lines up to 130 characters; helper
  `rel_parts` for the types of SymPy relations.
- D34: venv from the CPython managed by uv; bytecode compiled at install time (start of a
  calculation process from 4.997 s to 2.853 s, same Python).
- D35: `--install-lean` installs the toolchain before `lake update` and retries the rename that
  fails on the network drive; warning if elan or the Lean project sit on a network drive.
- D36 (security): `lean_check` rejects with `FORBIDDEN` code that runs programs or adds
  axioms, checks the axioms used (new field `axioms`) and at timeout kills the whole process
  tree. Lean is "available" only if Mathlib is compiled. Spec 3.3.0 synchronised to the dataset folder
  and to the project knowledge base.
- D37 (proposal to be confirmed): Lean and Mathlib on a local disk when the repo is on a network drive.
  `lean/lake-manifest.json` in the repo pins the 9 dependencies of Mathlib.
- LF line endings in all text files, as `.gitattributes` requires; `line-ending = "lf"` in ruff.
- Fixes found by the tests: limit with floating-point test point (the exact calculation
  hung), decimals with huge exponent rejected, `brentq` with valid tolerance, non-deprecated
  SymPy functions, `differential_evolution` with `rng` instead of `seed`.
- Command `--benchmark` (start-up, latency, timeout and restart of the calculation processes) and
  `docs/BENCHMARK.md` with the results.
- README rewritten for the working version.

## 0.0.4 - 2026-11-24

Spec 3.2.0 and test suite (TDD: tests before code).

- D24: `date_calc` with `recurring`, `convert_tz`, age in months and weeks, real hours between two instants.
- D25: `mcp` 2.3.0, server on `MCPServer` (formerly FastMCP).
- D26: Python environment on the local drive. `uv sync` measurement: `.venv` on the network drive 210.5 s the first time and 352.3 s with
  a ready cache; venv on the local drive 19.9 s with a ready cache.
- `pyproject.toml` (uv, hatchling, version read from `VERSION`), `.gitignore` for Lean.
- Tests in `tests/`: contract, security and fuzz, sandbox, calculation processes, all tools, real
  stdio and HTTP tests.
- Command `oce-math-mcp --sync-spec`: copies the spec to the dataset folder and rebuilds the spec document in the project knowledge base.

## 0.0.3 - 2026-11-23

Spec 3.1.0. No code.

- D22: 11 operations for 3D drawing in `geom3d_calc` (24 operations in total).
- D23: field `show_steps` in `math_eval`, `math_solve`, `number_theory`, `geom3d_calc`.
- Copies resynchronised to the dataset folder and to the project knowledge base.

## 0.0.2 - 2026-11-23

Final tool catalogue. No code.

- Decisions D13-D21 in `docs/DECISIONS.md`: 10 tools, `python_sandbox` as an AST interpreter,
  `lean_check` with 60 s timeout and Lean inside the repo, `eng_calc` in phase 2, holidays with
  4 October from 2026, people born on 29 February.
- New spec `docs/spec/SPEC_OCE_MATH_MCP_V3.md` and machine contract
  `docs/spec/oce_math_mcp_tools.json` (10 schemas, 127 operations counted from the file).
- Spec V2 marked as superseded; copies synchronised to the dataset folder and to the project knowledge base.

## 0.0.1 - 2026-11-23

Project start. No code.

- Created the repo `oce-math-mcp`, branch `main`.
- Copied the V2 spec into `docs/spec/` (identical to the file in the dataset folder,
  SHA-256 starting with `bc38931bec10d5d5ab4e`).
- Wrote the start-up discussion, the decision log, a new-session prompt, `README.md`, `CLAUDE.md`.
  (In 0.3.0 the V2 spec, the discussion and the prompt moved to the project knowledge base, D63.)
- Apache 2.0 licence like the other public repos of `oce-projects`.
- Repo published on GitHub as `oce-projects/oce-math-mcp` and cloned into the projects folder of
  DevAgentAi on the server.
