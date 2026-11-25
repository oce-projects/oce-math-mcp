"""number_theory: exact integers of any size within 10,000 digits (spec V3, section 5)."""

from __future__ import annotations

import itertools
import math
import re
from collections.abc import Callable
from typing import Any, cast

import sympy as sp
from sympy import ntheory as nt
from sympy.functions.combinatorial.numbers import jacobi_symbol, legendre_symbol, partition
from sympy.ntheory import digits
from sympy.ntheory.continued_fraction import (
    continued_fraction,
    continued_fraction_convergents,
    continued_fraction_iterator,
)
from sympy.ntheory.modular import crt as _crt
from sympy.solvers.diophantine import diophantine

from .. import limits
from ..envelope import Ctx
from ..errors import ToolError
from ..expr import Grammar
from ..expr import parse as parse_expr_safe
from ..fmt import int_text
from .common import SOLVE_FUNCS, expr_of, grammar, symbols_for

Op = Callable[[dict[str, Any], Ctx], dict[str, Any]]
OPS: dict[str, Op] = {}
_DEC = re.compile(r"^[+-]?\d+$")
_MAX_BITS = int(limits.INT_DIGITS_MAX * 3.3219280948873626) + 1


def op(name: str) -> Callable[[Op], Op]:
    def deco(fn: Op) -> Op:
        OPS[name] = fn
        return fn

    return deco


def handle(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    return OPS[args["operation"]](args, ctx)


def to_int(raw: Any, name: str) -> int:
    """Integer from the JSON: integer number or decimal string (also an integer expression, e.g. '2**127 - 1')."""
    if isinstance(raw, bool):
        raise ToolError("INVALID_ARGUMENT", f"'{name}' must be an integer.", "97")
    if isinstance(raw, int):
        value = raw
    elif isinstance(raw, float) and raw.is_integer():
        value = int(raw)
    elif isinstance(raw, str):
        text = raw.strip().replace("_", "")
        if len(text) > limits.INT_DIGITS_MAX + 1 and _DEC.match(text):
            raise ToolError(
                "LIMIT_EXCEEDED", f"'{name}' has more than {limits.INT_DIGITS_MAX} digits."
            )
        if _DEC.match(text):
            value = int(text)
        else:
            out = parse_expr_safe(text, Grammar(SOLVE_FUNCS, {}, free_symbols=False))
            if not getattr(out, "is_Integer", False):
                raise ToolError(
                    "INVALID_ARGUMENT",
                    f"'{name}' is not an integer: {text[:60]!r}.",
                    '"2**127 - 1"',
                )
            value = int(out)
    else:
        raise ToolError(
            "INVALID_ARGUMENT",
            f"'{name}' must be an integer or a string of digits.",
            '"123456789012345678901"',
        )
    if value.bit_length() > _MAX_BITS:
        raise ToolError("LIMIT_EXCEEDED", f"'{name}' has more than {limits.INT_DIGITS_MAX} digits.")
    return value


def _get(args: dict[str, Any], key: str, example: str) -> int:
    if args.get(key) is None:
        raise ToolError(
            "INVALID_ARGUMENT", f"'{args['operation']}' needs the field '{key}'.", example
        )
    return to_int(args[key], key)


def _out(n: int) -> int | str:
    """Integer for the JSON; beyond 10,000 digits it becomes a text summary."""
    return n if n.bit_length() <= _MAX_BITS else int_text(n)


def _positive(n: int, name: str) -> None:
    if n < 1:
        raise ToolError("INVALID_ARGUMENT", f"'{name}' must be a positive integer.", "12")


# ------------------------------------------------------------------ primes


@op("is_prime")
def _is_prime(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    n = _get(args, "n", '"170141183460469231731687303715884105727"')
    res = bool(sp.isprime(n))
    method = "deterministic Miller-Rabin" if n < 2**64 else "BPSW (Baillie-PSW)"
    ctx.step(f"n = {int_text(n)}.")
    ctx.step(f"Primality test: {method}.")
    if not res and n > 1:
        small = next((p for p in sp.primerange(2, 10**5) if n % p == 0 and p != n), None)
        if small is not None:
            ctx.step(f"Check: {small} divides n, so n is not prime.")
    ctx.formatted = f"{int_text(n)} {'is' if res else 'is not'} prime."
    return {"is_prime": res, "method": method}


@op("factorint")
def _factorint(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    n = _get(args, "n", "360")
    if n == 0:
        raise ToolError("MATH_ERROR", "0 has no factorisation.")
    f = sp.factorint(n)
    keys = sorted(f)
    product = math.prod(p**e for p, e in f.items() if p > 0) * (-1 if n < 0 else 1)
    verified = product == n
    text = " · ".join(f"{p}^{f[p]}" if f[p] > 1 else str(p) for p in keys) or "1"
    ctx.step(f"Factorising {int_text(n)} (trial division, Pollard rho, ECM).")
    ctx.step(f"Factors: {text}.")
    ctx.step(f"Back-check: multiplying the factors gives back {int_text(product)}.")
    ctx.formatted = f"{int_text(n)} = {text}"
    return {
        "factors": {str(p): int(e) for p, e in sorted(f.items())},
        "factorization": text,
        "verified": verified,
    }


@op("divisors")
def _divisors(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    n = abs(_get(args, "n", "28"))
    _positive(n, "n")
    limit = int(args.get("limit", 1000))
    count = int(sp.divisor_count(n))
    total = int(sp.divisor_sigma(n, 1))
    divs = (
        list(itertools.islice(sp.divisors(n, generator=True), limit))
        if count > limit
        else sp.divisors(n)
    )
    truncated = count > limit
    if truncated:
        ctx.warn(f"Showing {limit} divisors out of {count}: raise 'limit' up to 10000.")
    ctx.formatted = f"{n} has {count} divisors, sum {total}"
    return {
        "divisors": [int(d) for d in divs],
        "count": count,
        "sum": total,
        "truncated": truncated,
    }


def _simple(
    name: str, fn: Callable[[int], Any], label: str, check: Callable[[int], None] | None = None
) -> None:
    def run(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
        n = _get(args, "n", "36")
        if check:
            check(n)
        value = int(fn(n))
        ctx.formatted = f"{label}({n}) = {value}"
        return {"value": _out(value)}

    OPS[name] = run


def _n_positive(n: int) -> None:
    _positive(n, "n")


def _above_two(n: int) -> None:
    if n <= 2:
        raise ToolError("MATH_ERROR", "There is no prime less than 2.")


def _non_negative(n: int) -> None:
    if n < 0:
        raise ToolError("INVALID_ARGUMENT", "n must be >= 0.", "100")


_simple("totient", sp.totient, "φ", _n_positive)
_simple("mobius", sp.mobius, "μ", _n_positive)
_simple("next_prime", sp.nextprime, "next prime after ")
_simple("prev_prime", sp.prevprime, "previous prime before ", _above_two)
_simple("nth_prime", sp.prime, "p", _n_positive)
_simple("prime_pi", sp.primepi, "π", None)
_simple("partitions", partition, "p", _non_negative)


@op("primerange")
def _primerange(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    a = _get(args, "n", "10")
    b = _get(args, "m", "30")
    limit = int(args.get("limit", 1000))
    primes = [int(p) for p in itertools.islice(sp.primerange(a, b), limit + 1)]
    truncated = len(primes) > limit
    primes = primes[:limit]
    if truncated:
        ctx.warn(f"List stopped at {limit} primes: raise 'limit' or narrow the interval.")
    ctx.formatted = f"{len(primes)} primes in [{a}, {b})"
    return {"primes": primes, "count": len(primes), "truncated": truncated}


def _values(args: dict[str, Any]) -> list[int]:
    raw = args.get("values")
    vals = (
        [to_int(v, "values") for v in raw]
        if raw
        else [_get(args, "n", "84"), _get(args, "m", "36")]
    )
    if len(vals) > limits.LIST_MAX:
        raise ToolError("LIMIT_EXCEEDED", f"At most {limits.LIST_MAX} values.")
    return vals


@op("gcd")
def _gcd(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    vals = _values(args)
    g = math.gcd(*vals)
    ctx.formatted = f"GCD = {g}"
    return {"value": _out(g)}


@op("lcm")
def _lcm(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    vals = _values(args)
    v = math.lcm(*vals)
    ctx.formatted = f"LCM = {int_text(v)}"
    return {"value": _out(v)}


@op("gcd_ext")
def _gcd_ext(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    a = _get(args, "n", "240")
    b = _get(args, "m", "46")
    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1
    ctx.step(f"Extended Euclidean algorithm on {a} and {b}.")
    rounds = 0
    while r != 0:
        q = old_r // r
        if rounds < 40:
            ctx.step(f"Euclid: {old_r} = {q}·{r} + {old_r - q * r}.")
        rounds += 1
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t
    if old_r < 0:
        old_r, old_s, old_t = -old_r, -old_s, -old_t
    if rounds > 40:
        ctx.step(f"... {rounds - 40} more divisions.")
    verified = a * old_s + b * old_t == old_r
    ctx.step(f"GCD = {old_r}; Bézout coefficients x = {old_s}, y = {old_t}.")
    ctx.step(f"Back-check: {a}·({old_s}) + {b}·({old_t}) = {a * old_s + b * old_t}.")
    ctx.formatted = f"GCD({a}, {b}) = {old_r} = {a}·({old_s}) + {b}·({old_t})"
    return {"gcd": _out(old_r), "x": _out(old_s), "y": _out(old_t), "verified": verified}


# ------------------------------------------------------------------ modular


def _modulus(args: dict[str, Any]) -> int:
    m = _get(args, "m", "1000000007")
    if m < 1:
        raise ToolError("INVALID_ARGUMENT", "The modulus 'm' must be >= 1.", "1000000007")
    return m


@op("mod_pow")
def _mod_pow(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    base = _get(args, "n", "2")
    exp = _get(args, "k", "1000000000000000000")
    m = _modulus(args)
    try:
        v = pow(base, exp, m)
    except ValueError:
        raise ToolError(
            "MATH_ERROR", f"{base} is not invertible modulo {m}: negative exponent impossible."
        ) from None
    ctx.step("Fast exponentiation (repeated squaring) modulo m.")
    ctx.formatted = f"{base}^{exp} mod {m} = {v}"
    return {"value": v}


@op("mod_inv")
def _mod_inv(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    n = _get(args, "n", "17")
    m = _modulus(args)
    try:
        v = pow(n, -1, m)
    except ValueError:
        raise ToolError(
            "MATH_ERROR", f"{n} is not invertible modulo {m}: GCD({n}, {m}) = {math.gcd(n, m)}."
        ) from None
    ctx.step(
        f"Inverse with the extended Euclidean algorithm; check: {n}·{v} mod {m} = {(n * v) % m}."
    )
    ctx.formatted = f"{n}^-1 mod {m} = {v}"
    return {"value": v, "verified": (n * v) % m == 1 % m}


@op("sqrt_mod")
def _sqrt_mod(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    n = _get(args, "n", "2")
    m = _modulus(args)
    raw = cast(list[int] | None, nt.sqrt_mod(n, m, all_roots=True))
    roots = sorted(int(r) for r in (raw or []))
    if len(roots) > limits.LIST_MAX:
        roots = roots[: limits.LIST_MAX]
        ctx.warn(f"List stopped at {limits.LIST_MAX} roots.")
    ctx.formatted = f"x^2 ≡ {n} (mod {m}): {roots or 'no solution'}"
    return {"roots": roots, "verified": all(r * r % m == n % m for r in roots)}


@op("discrete_log")
def _discrete_log(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    value = _get(args, "n", "13")
    base = _get(args, "k", "3")
    m = _modulus(args)
    x = int(nt.discrete_log(m, value, base))
    ctx.formatted = f"{base}^{x} ≡ {value} (mod {m})"
    return {"value": x, "verified": pow(base, x, m) == value % m}


@op("primitive_root")
def _primitive_root(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    n = _get(args, "n", "17")
    _positive(n, "n")
    g = nt.primitive_root(n)
    if g is None:
        raise ToolError(
            "MATH_ERROR", f"{n} has no primitive roots (1, 2, 4, p^k or 2p^k are needed)."
        )
    ctx.formatted = f"Smallest primitive root modulo {n}: {g}"
    return {"value": int(g)}


@op("multiplicative_order")
def _order(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    n = _get(args, "n", "3")
    m = _modulus(args)
    if math.gcd(n, m) != 1:
        raise ToolError("MATH_ERROR", f"{n} and {m} are not coprime: the order does not exist.")
    k = int(nt.n_order(n, m))
    ctx.formatted = f"Order of {n} modulo {m}: {k}"
    return {"value": k, "verified": pow(n, k, m) == 1 % m}


@op("legendre")
def _legendre(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    n = _get(args, "n", "2")
    p = _get(args, "m", "7")
    if p < 3 or not sp.isprime(p):
        raise ToolError("INVALID_ARGUMENT", "The Legendre symbol needs m an odd prime.", "7")
    v = int(legendre_symbol(n % p, p))
    ctx.step(f"Euler's criterion: {n}^(({p}-1)/2) mod {p} = {pow(n, (p - 1) // 2, p)}.")
    ctx.formatted = f"({n}/{p}) = {v}"
    return {"value": v}


@op("jacobi")
def _jacobi(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    n = _get(args, "n", "2")
    m = _get(args, "m", "15")
    if m < 1 or m % 2 == 0:
        raise ToolError("INVALID_ARGUMENT", "The Jacobi symbol needs m odd and positive.", "15")
    v = int(jacobi_symbol(n % m, m))
    ctx.formatted = f"({n}/{m}) = {v}"
    return {"value": v}


@op("crt")
def _crt_op(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    vals = [to_int(v, "values") for v in args.get("values") or []]
    mods = [to_int(v, "moduli") for v in args.get("moduli") or []]
    if not vals or len(vals) != len(mods):
        raise ToolError(
            "INVALID_ARGUMENT",
            "'values' and 'moduli' of the same length are needed.",
            '{"values": [2, 3, 2], "moduli": [3, 5, 7]}',
        )
    if any(m < 1 for m in mods):
        raise ToolError("INVALID_ARGUMENT", "The moduli must be >= 1.", "[3, 5, 7]")
    res = _crt(mods, vals)
    if res is None:
        raise ToolError(
            "MATH_ERROR",
            "The system of congruences has no solution (moduli not coprime and incompatible remainders).",
        )
    v, mod = int(res[0]), int(res[1])
    for r, m in zip(vals, mods, strict=True):
        ctx.step(f"x ≡ {r} (mod {m}).")
    ctx.step(f"Chinese remainder theorem: x = {v} (mod {mod}).")
    ctx.step("Back-check: " + ", ".join(f"{v} mod {m} = {v % m}" for m in mods) + ".")
    ctx.formatted = f"x ≡ {v} (mod {mod})"
    return {
        "value": v,
        "modulus": mod,
        "verified": all(v % m == r % m for r, m in zip(vals, mods, strict=True)),
    }


# ------------------------------------------------------------------ other


@op("diophantine")
def _diophantine(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    names = list(args.get("variables") or [])
    g = grammar(names)
    eq = expr_of(args.get("expr") or "", g) if args.get("expr") else None
    if eq is None:
        raise ToolError("INVALID_ARGUMENT", "'expr' is needed with the equation.", "3*x + 5*y = 1")
    xs = symbols_for(names) or sorted(eq.free_symbols, key=lambda s: s.name)
    sols = diophantine(sp.expand(eq), syms=xs)
    limit = int(args.get("limit", 1000))
    out = []
    verified = True
    for sol in itertools.islice(sols, limit):
        out.append({str(v): str(s) for v, s in zip(xs, sol, strict=True)})
        if sp.simplify(eq.subs(dict(zip(xs, sol, strict=True)))) != 0:
            verified = False
    ctx.formatted = f"{len(out)} families of solutions" if out else "No integer solution."
    if out and any("t_" in v for s in out for v in s.values()):
        ctx.warn("t_0, t_1, ... are free integer parameters.")
    return {"solutions": out, "verified": verified}


@op("continued_fraction")
def _continued_fraction(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    limit = min(int(args.get("limit", 1000)), 1000)
    if args.get("expr"):
        x = parse_expr_safe(args["expr"], Grammar(SOLVE_FUNCS, {}, free_symbols=False))
    else:
        x = sp.Rational(_get(args, "n", "415"), _get(args, "m", "93"))
    if not x.is_real:
        raise ToolError("INVALID_ARGUMENT", "A real number is needed.", '"415/93"')
    periodic = False
    period: list[int] = []
    try:
        res = continued_fraction(x)
        if res and isinstance(res[-1], list):
            periodic = True
            period = [int(v) for v in res[-1]]
            terms = [int(v) for v in res[:-1]] + period
        else:
            terms = [int(v) for v in res]
    except (ValueError, TypeError, NotImplementedError):
        n_terms = min(limit, 60)
        terms = [int(v) for v in itertools.islice(continued_fraction_iterator(x), n_terms)]
        ctx.warn(f"Number neither rational nor quadratic: showing the first {len(terms)} terms.")
    terms = terms[:limit]
    convs = [str(c) for c in itertools.islice(continued_fraction_convergents(terms), 20)]
    ctx.formatted = f"[{terms[0]}; {', '.join(map(str, terms[1:]))}]" + (
        " (periodic)" if periodic else ""
    )
    return {"terms": terms, "periodic": periodic, "period": period, "convergents": convs}


@op("digits")
def _digits(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    n = abs(_get(args, "n", "1267650600228229401496703205376"))
    base = to_int(args["k"], "k") if args.get("k") is not None else 10
    if not 2 <= base <= 1_000_000:
        raise ToolError("INVALID_ARGUMENT", "The base 'k' must be between 2 and 1,000,000.", "10")
    ds = [int(c) for c in str(n)] if base == 10 else [int(d) for d in digits(n, base)[1:]]
    ctx.formatted = f"{len(ds)} digits in base {base}, sum {sum(ds)}"
    return {"digits": ds, "digit_sum": sum(ds), "count": len(ds), "base": base}
