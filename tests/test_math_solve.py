"""math_solve: symbolic with self-verification."""

from __future__ import annotations

import math

import sympy as sp

from tests.conftest import call, err, ok

x, y, z, t, s, n = sp.symbols("x y z t s n")


def same(a: str, b: str) -> bool:
    """Symbolic comparison in the tests (objects built by the test code, not by the model)."""
    loc = {k: v for k, v in vars(sp).items() if not k.startswith("_")}
    loc.update({"x": x, "y": y, "z": z, "t": t, "s": s, "n": n})
    return sp.simplify(eval(a, {"__builtins__": {}}, loc) - eval(b, {"__builtins__": {}}, loc)) == 0


def solve(**kw: object) -> dict:
    return ok("math_solve", **kw)


def test_solve_quadratic() -> None:
    r = solve(operation="solve", expr="x**2 - 5*x + 6 = 0")
    assert sorted(r["solutions"]) == ["2", "3"]
    assert r["count"] == 2
    assert r["verified"] is True


def test_solve_quadratic_steps() -> None:
    r = solve(operation="solve", expr="x**2 - 5*x + 6 = 0", show_steps=True)
    text = " ".join(st["text"] for st in r["steps"])
    assert "Discriminant" in text
    assert "check" in r["steps"][-1]["text"].lower()


def test_solve_domain() -> None:
    assert solve(operation="solve", expr="x**2 + 1 = 0")["solutions"] == []
    assert sorted(solve(operation="solve", expr="x**2 + 1 = 0", domain="complex")["solutions"]) == [
        "-I",
        "I",
    ]


def test_solve_system() -> None:
    r = solve(operation="solve_system", expr="x + y = 3; x - y = 1", variables=["x", "y"])
    assert r["solutions"] == [{"x": "2", "y": "1"}]
    assert r["verified"]


def test_inequality() -> None:
    r = solve(operation="inequality", expr="x**2 - 4 > 0", variables=["x"])
    assert "-oo" in r["solution_set"] and "oo" in r["solution_set"]
    assert r["verified"]


def test_diff_and_order() -> None:
    assert same(
        solve(operation="diff", expr="sin(x)*x", variables=["x"])["derivative"], "x*cos(x) + sin(x)"
    )
    assert same(
        solve(operation="diff", expr="x**4", variables=["x"], order=2)["derivative"], "12*x**2"
    )


def test_diff_steps_name_the_rule() -> None:
    r = solve(operation="diff", expr="sin(x**2)", variables=["x"], show_steps=True)
    assert "chain" in " ".join(st["text"] for st in r["steps"]).lower()


def test_partial_diff() -> None:
    r = solve(operation="partial_diff", expr="x**2*y**3", variables=["x", "y"])
    assert same(r["derivative"], "6*x*y**2")


def test_integrate_indefinite_and_definite() -> None:
    r = solve(operation="integrate", expr="x**2", variables=["x"])
    assert same(r["integral"], "x**3/3") and r["verified"] and r["definite"] is False
    r = solve(operation="integrate", expr="x**2", variables=["x"], bounds=[0, 1])
    assert r["integral"] == "1/3" and r["verified"]


def test_integrate_improper_gaussian() -> None:
    r = solve(operation="integrate", expr="exp(-x**2)", variables=["x"], bounds=["-oo", "oo"])
    assert r["integral"] == "sqrt(pi)"
    assert abs(r["numeric"] - math.sqrt(math.pi)) < 1e-12


def test_integrate_by_parts_steps() -> None:
    r = solve(operation="integrate", expr="x*exp(x)", variables=["x"], show_steps=True)
    assert "parts" in " ".join(st["text"] for st in r["steps"]).lower()
    assert r["verified"]


def test_multi_integrate_cartesian_polar_spherical() -> None:
    assert (
        solve(operation="multi_integrate", expr="x*y", bounds=[["x", 0, 1], ["y", 0, 2]])[
            "integral"
        ]
        == "1"
    )
    assert (
        solve(
            operation="multi_integrate",
            expr="1",
            bounds=[["r", 0, 1], ["theta", 0, "2*pi"]],
            coords="polar",
        )["integral"]
        == "pi"
    )
    r = solve(
        operation="multi_integrate",
        expr="1",
        bounds=[["r", 0, 1], ["theta", 0, "pi"], ["phi", 0, "2*pi"]],
        coords="spherical",
    )
    assert r["integral"] == "4*pi/3"


def test_line_integral_work() -> None:
    r = solve(
        operation="line_integral",
        expr="x=cos(t); y=sin(t); z=0",
        variables=["t"],
        bounds=[0, "2*pi"],
        vector_field=["-y", "x", "0"],
    )
    assert r["integral"] == "2*pi"


def test_surface_integral_flux_unit_sphere() -> None:
    r = solve(
        operation="surface_integral",
        expr="x=sin(u)*cos(v); y=sin(u)*sin(v); z=cos(u)",
        bounds=[["u", 0, "pi"], ["v", 0, "2*pi"]],
        vector_field=["x", "y", "z"],
    )
    assert r["integral"] == "4*pi"


def test_limits() -> None:
    assert solve(operation="limit", expr="sin(x)/x", variables=["x"], bounds=[0])["limit"] == "1"
    assert solve(operation="limit", expr="1/x", variables=["x"], bounds=[0, "+"])["limit"] == "oo"
    assert (
        solve(operation="limit", expr="(1 + 1/n)**n", variables=["n"], bounds=["oo"])["limit"]
        == "E"
    )


def test_series() -> None:
    r = solve(operation="series", expr="exp(x)", variables=["x"], order=4)
    assert same(r["polynomial"], "1 + x + x**2/2 + x**3/6")


def test_function_study() -> None:
    r = solve(operation="function_study", expr="x**3 - 3*x", variables=["x"])
    kinds = {cp["x"]: cp["type"] for cp in r["critical_points"]}
    assert kinds == {"-1": "max", "1": "min"}
    assert [p["x"] for p in r["inflection_points"]] == ["0"]
    assert sorted(r["zeros"]) == sorted(["0", "-sqrt(3)", "sqrt(3)"])


def test_simplify_factor_expand_apart() -> None:
    assert solve(operation="simplify", expr="sin(x)**2 + cos(x)**2")["expression"] == "1"
    assert same(solve(operation="factor", expr="x**2 - 1")["expression"], "(x - 1)*(x + 1)")
    assert same(
        solve(operation="expand", expr="(x + 1)**3")["expression"], "x**3 + 3*x**2 + 3*x + 1"
    )
    r = solve(operation="partial_fractions", expr="1/(x**2 - 1)", variables=["x"])
    assert same(r["expression"], "1/(2*(x - 1)) - 1/(2*(x + 1))") and r["verified"]


def test_vector_calculus() -> None:
    assert solve(operation="gradient", expr="x**2 + y**2", variables=["x", "y"])["gradient"] == [
        "2*x",
        "2*y",
    ]
    assert (
        solve(operation="divergence", vector_field=["x", "y", "z"], variables=["x", "y", "z"])[
            "divergence"
        ]
        == "3"
    )
    assert solve(operation="curl", vector_field=["-y", "x", "0"], variables=["x", "y", "z"])[
        "curl"
    ] == ["0", "0", "2"]
    assert (
        solve(operation="laplacian", expr="x**2 + y**2 + z**2", variables=["x", "y", "z"])[
            "laplacian"
        ]
        == "6"
    )
    r = solve(operation="jacobian", vector_field=["x*y", "x + y"], variables=["x", "y"])
    assert r["jacobian"] == [["y", "x"], ["1", "1"]]
    assert same(r["determinant"], "y - x")


def test_hessian_classification() -> None:
    r = solve(operation="hessian", expr="x**2 + y**2", variables=["x", "y"])
    assert r["critical_points"] == [{"point": {"x": "0", "y": "0"}, "type": "minimum"}]
    r = solve(operation="hessian", expr="x**2 - y**2", variables=["x", "y"])
    assert r["critical_points"][0]["type"] == "saddle"


def test_lagrange() -> None:
    r = solve(
        operation="lagrange_optimize",
        expr="x + y",
        constraints=["x**2 + y**2 = 1"],
        variables=["x", "y"],
    )
    assert r["maximum"]["value"] == "sqrt(2)"
    assert r["minimum"]["value"] == "-sqrt(2)"


def test_ode_second_order_with_ics() -> None:
    r = solve(
        operation="ode", expr="y'' + y = 0", variables=["y", "x"], ics={"y(0)": 0, "y'(0)": 1}
    )
    assert r["solution"] == "sin(x)"
    assert r["method"] == "symbolic" and r["verified"]


def test_ode_first_order() -> None:
    r = solve(operation="ode", expr="y' = y", variables=["y", "x"], ics={"y(0)": 1})
    assert r["solution"] == "exp(x)"


def test_laplace_pair() -> None:
    r = solve(operation="laplace", expr="exp(-2*t)", variables=["t", "s"])
    assert same(r["transform"], "1/(s + 2)") and r["verified"]
    r = solve(operation="inv_laplace", expr="1/(s**2 + 1)", variables=["s", "t"])
    assert "sin(t)" in r["transform"]


def test_fft_dominant_frequency() -> None:
    fs, f0 = 64, 5
    samples = [math.sin(2 * math.pi * f0 * k / fs) for k in range(64)]
    r = solve(operation="fourier_fft", numeric_samples=samples, bounds=[fs])
    assert r["dominant_frequency"] == 5.0


def test_fourier_series_symbolic() -> None:
    r = solve(operation="fourier_fft", expr="x", variables=["x"], bounds=["-pi", "pi"], order=3)
    assert r["bn"] == ["2", "-1", "2/3"]
    assert r["a0"] == "0"


def test_riccati_lqr_double_integrator() -> None:
    r = solve(
        operation="riccati_lqr",
        matrix_a=[[0, 1], [0, 0]],
        matrix_b=[[0], [1]],
        matrix_q=[[1, 0], [0, 1]],
        matrix_r=[[1]],
    )
    r3 = math.sqrt(3)
    expected_p = [[r3, 1.0], [1.0, r3]]
    for i in range(2):
        for j in range(2):
            assert abs(r["P"][i][j] - expected_p[i][j]) < 1e-8
    assert abs(r["K"][0][1] - r3) < 1e-8
    assert r["verified"]


def test_matrix_ops() -> None:
    a = [[1, 2], [3, 4]]
    assert solve(operation="matrix", matrix_a=a, matrix_op="det")["value"] == "-2"
    assert solve(operation="matrix", matrix_a=a, matrix_op="inv")["matrix"] == [
        ["-2", "1"],
        ["3/2", "-1/2"],
    ]
    assert solve(operation="matrix", matrix_a=[[1, 2], [2, 4]], matrix_op="rank")["value"] == "1"
    assert solve(operation="matrix", matrix_a=[[2, 0], [0, 3]], matrix_op="eigenvals")[
        "eigenvalues"
    ] == {"2": 1, "3": 1}
    r = solve(operation="matrix", matrix_a=[[4, 2], [2, 3]], matrix_op="cholesky")
    assert r["verified"]
    r = solve(operation="matrix", matrix_a=[[3.0, 0.0], [0.0, 4.0]], matrix_op="svd")
    assert sorted(float(v) for v in r["S"]) == [3.0, 4.0]
    for op in ("lu", "qr", "pinv", "rref", "transpose", "cofactor"):
        assert solve(operation="matrix", matrix_a=a, matrix_op=op)["verified"], op
    r = solve(operation="matrix", matrix_a=a, matrix_b=[[5], [6]], matrix_op="solve")
    assert r["solution"] == [["-4"], ["9/2"]]


def test_matrix_size_limit() -> None:
    big = [[1] * 51 for _ in range(51)]
    assert err("math_solve", operation="matrix", matrix_a=big, matrix_op="det") == "LIMIT_EXCEEDED"


def test_optimize_num() -> None:
    r = solve(operation="optimize_num", expr="(x - 1)**2 + (y - 2)**2", variables=["x", "y"])
    assert abs(r["x"]["x"] - 1) < 1e-6 and abs(r["x"]["y"] - 2) < 1e-6


def test_complex_ops() -> None:
    r = solve(operation="complex_ops", expr="3 + 4*I")
    assert r["modulus"] == "5"
    assert abs(r["argument_deg"] - math.degrees(math.atan2(4, 3))) < 1e-9
    assert r["conjugate"] == "3 - 4*I"


def test_summation() -> None:
    r = solve(operation="summation", expr="k", variables=["k"], bounds=[1, "n"])
    assert same(r["sum"], "n*(n + 1)/2") and r["verified"]
    assert (
        solve(operation="summation", expr="1/k**2", variables=["k"], bounds=[1, "oo"])["sum"]
        == "pi**2/6"
    )


def test_recurrence_fibonacci() -> None:
    r = solve(
        operation="recurrence",
        expr="a(n) = a(n-1) + a(n-2)",
        variables=["a", "n"],
        ics={"a(0)": 0, "a(1)": 1},
    )
    assert r["first_terms"][:10] == ["0", "1", "1", "2", "3", "5", "8", "13", "21", "34"]
    assert r["verified"]


def test_polynomial_ops() -> None:
    p = "x**3 - 6*x**2 + 11*x - 6"
    assert solve(operation="polynomial", expr=p, variables=["x"], poly_op="roots")["value"] == {
        "1": 1,
        "2": 1,
        "3": 1,
    }
    assert (
        solve(
            operation="polynomial", expr="x**2 - 2*x + 1", variables=["x"], poly_op="discriminant"
        )["value"]
        == "0"
    )
    v = solve(operation="polynomial", expr=p, variables=["x"], poly_op="vieta")["value"]
    assert v["e1"] == "6" and v["e3"] == "6"
    g = solve(
        operation="polynomial", expr="x**2 - 1; x**2 + 2*x + 1", variables=["x"], poly_op="gcd"
    )["value"]
    assert g == "x + 1"


def test_steps_present_for_any_operation() -> None:
    r = solve(operation="matrix", matrix_a=[[1, 2], [3, 4]], matrix_op="det", show_steps=True)
    assert len(r["steps"]) >= 2


def test_unknown_function_is_forbidden() -> None:
    resp = call("math_solve", operation="simplify", expr="system(1)")
    assert resp["error"]["code"] == "FORBIDDEN"
