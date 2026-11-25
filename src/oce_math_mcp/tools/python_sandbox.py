"""python_sandbox: AST interpreter for a subset of Python (decision D13, spec V3 section 8).

No exec/eval/compile: the code is read with ast.parse and every node is executed by this
module. Defences, from the first to the last:

1. static check: forbidden constructs (class, with, yield, global, async), names with "__",
   attributes that start with "_" or are on a deny list (format, ctypes, tofile, sympify, ...);
2. imports only from an allowed list; modules arrive as ModuleProxy (for `time` and `datetime`
   only the attributes of _ALLOWED_ATTRS, D40);
3. reduced builtins; powers, sequence multiplications and list conversions with limits;
4. internal deadline (TIMEOUT) and, outside of here, the computing process that gets killed;
5. library stdout captured; no files, no network.
"""

from __future__ import annotations

import ast
import contextlib
import decimal
import fractions
import importlib
import io
import itertools
import operator
import random
import time
import types
from collections.abc import Callable, Iterator
from typing import Any

from .. import limits
from ..envelope import Ctx
from ..errors import ToolError
from ..harden import harden

ALLOWED_TOP = frozenset(
    {
        "math",
        "cmath",
        "fractions",
        "decimal",
        "itertools",
        "functools",
        "collections",
        "statistics",
        "random",
        "heapq",
        "bisect",
        "sympy",
        "numpy",
        "scipy",
        "mpmath",
        "time",  # D40: clocks only
        "datetime",  # D40
    }
)
# D40: partially allowed modules. time has no sleep (it would stall the interpreter past the
# deadline) nor get_clock_info; datetime has no datetime_CAPI and no sys.
_ALLOWED_ATTRS: dict[str, frozenset[str]] = {
    "time": frozenset(
        {
            "time", "time_ns", "perf_counter", "perf_counter_ns", "monotonic", "monotonic_ns",
            "process_time", "process_time_ns", "localtime", "gmtime", "strftime", "ctime",
            "asctime", "mktime", "struct_time",
        }
    ),
    "datetime": frozenset(
        {"date", "datetime", "time", "timedelta", "timezone", "tzinfo", "MINYEAR", "MAXYEAR", "UTC"}
    ),
}  # fmt: skip
_ALLOWED_SUB = {
    "collections": {"abc"},
    "numpy": {"linalg", "random", "fft", "polynomial", "emath"},
    "scipy": {
        "optimize",
        "integrate",
        "special",
        "linalg",
        "stats",
        "interpolate",
        "fft",
        "spatial",
        "constants",
        "signal",
        "sparse",
    },
}
_SYMPY_BLOCKED_SUB = frozenset(
    {
        "parsing",
        "utilities",
        "plotting",
        "printing",
        "external",
        "testing",
        "interactive",
        "codegen",
        "benchmarks",
        "conftest",
        "this",
        "release",
        "multipledispatch",
        "abc",
    }
)

BLOCKED_ATTRS = frozenset(
    {
        # formatting that reads attributes
        "format",
        "format_map",
        # numpy: files, raw memory, pickle
        "ctypes",
        "ctypeslib",
        "tofile",
        "fromfile",
        "fromregex",
        "dump",
        "dumps",
        "load",
        "loads",
        "save",
        "savez",
        "savez_compressed",
        "savetxt",
        "loadtxt",
        "genfromtxt",
        "memmap",
        "DataSource",
        "lib",
        "f2py",
        "distutils",
        "testing",
        "show_config",
        "show_runtime",
        "save_npz",
        "load_npz",
        "LowLevelCallable",
        # scipy
        "io",
        "datasets",
        "misc",
        # sympy: text -> code, printing to file, injection into globals
        "sympify",
        "parse_expr",
        "lambdify",
        "preview",
        "var",
        "init_session",
        "init_printing",
        "autowrap",
        "ufuncify",
        "test",
        "doctest",
        # mpmath
        "identify",
        "cplot",
        "splot",
        # collections.namedtuple usa exec
        "namedtuple",
        # generators, frames, code
        "gi_frame",
        "gi_code",
        "gi_yieldfrom",
        "gi_running",
        "cr_frame",
        "ag_frame",
        "f_globals",
        "f_locals",
        "f_back",
        "f_builtins",
        "f_code",
        "tb_frame",
        "tb_next",
        "mro",
        "system",
        "popen",
        "spawn",
        "eval",
        "exec",
        "compile",
        "open",
        "input",
        "breakpoint",
    }
)

FORBIDDEN_NAMES = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "open",
        "globals",
        "locals",
        "getattr",
        "setattr",
        "delattr",
        "hasattr",
        "type",
        "breakpoint",
        "input",
        "help",
        "exit",
        "quit",
        "memoryview",
        "object",
        "super",
        "classmethod",
        "staticmethod",
        "property",
        "dir",
    }
)

# D40: forbidden only as builtins. `def f(vars)` or `vars = [...]` are fine; `vars()` without the
# program binding the name stays FORBIDDEN (and the builtin is not among the sandbox ones anyway).
BUILTIN_ONLY_NAMES = frozenset({"vars"})

_FORBIDDEN_NODES: dict[str, str] = {
    "ClassDef": "class",
    "AsyncFunctionDef": "async def",
    "AsyncFor": "async for",
    "AsyncWith": "async with",
    "Await": "await",
    "Yield": "yield",
    "YieldFrom": "yield from",
    "Nonlocal": "nonlocal",
    "With": "with",
    "Match": "match",
    "TryStar": "except*",
}

_BIN: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
    ast.BitAnd: operator.and_,
    ast.MatMult: operator.matmul,
}
_CMP: dict[type, Callable[[Any, Any], Any]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}
_UNARY: dict[type, Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Not: operator.not_,
    ast.Invert: operator.invert,
}


class _Control(BaseException):
    """Control signals of the interpreter: the user cannot catch them."""


class _Break(_Control):
    pass


class _Continue(_Control):
    pass


class _Return(_Control):
    def __init__(self, value: Any) -> None:
        super().__init__()
        self.value = value


_UNCATCHABLE = (
    ToolError,
    _Control,
    RecursionError,
    MemoryError,
    KeyboardInterrupt,
    SystemExit,
    GeneratorExit,
)


def _forbid(msg: str, line: int | None = None) -> ToolError:
    where = f" (line {line})" if line else ""
    return ToolError("FORBIDDEN", f"{msg}{where}.", "import math; math.comb(10, 3)")


def _limit(msg: str) -> ToolError:
    return ToolError("LIMIT_EXCEEDED", msg)


# ------------------------------------------------------------------ moduli


def _module_allowed(fullname: str) -> bool:
    parts = fullname.split(".")
    if parts[0] not in ALLOWED_TOP:
        return False
    if len(parts) == 1:
        return True
    if parts[0] == "sympy":
        return parts[1] not in _SYMPY_BLOCKED_SUB and not any(p.startswith("_") for p in parts)
    return parts[1] in _ALLOWED_SUB.get(parts[0], set()) and not any(
        p.startswith("_") for p in parts[2:]
    )


def _check_attr(name: str, line: int | None = None) -> None:
    if name.startswith("_") or name in BLOCKED_ATTRS or name.startswith("plot"):
        raise _forbid(f"Attribute '{name}' not allowed", line)


class ModuleProxy:
    """Module as seen by the sandbox: only allowed attributes, no assignments."""

    def __init__(self, module: types.ModuleType) -> None:
        object.__setattr__(self, "_m", module)

    def _get(self, name: str) -> Any:
        _check_attr(name)
        mod: types.ModuleType = object.__getattribute__(self, "_m")
        allowed = _ALLOWED_ATTRS.get(mod.__name__)
        if allowed is not None and name not in allowed:
            raise _forbid(
                f"{mod.__name__}.{name} not allowed (allowed: {', '.join(sorted(allowed))})"
            )
        try:
            value = getattr(mod, name)
        except AttributeError:
            full = f"{mod.__name__}.{name}"
            if _module_allowed(full):
                value = importlib.import_module(full)
            else:
                raise
        return wrap_value(value)

    def __setattr__(self, name: str, value: Any) -> None:
        raise _forbid("Modules cannot be modified")

    def __repr__(self) -> str:
        return f"<module '{object.__getattribute__(self, '_m').__name__}'>"


class SingletonProxy:
    """`sympy.S` as seen by the sandbox (D38): constants (`S.Reals`, `S.Half`) yes, `S('text')` no."""

    def __init__(self, registry: Any) -> None:
        object.__setattr__(self, "_r", registry)

    def __getattr__(self, name: str) -> Any:
        import sympy as sp

        _check_attr(name)
        value = getattr(object.__getattribute__(self, "_r"), name)
        if not isinstance(value, sp.Basic):
            raise _forbid(f"S.{name} not allowed: only SymPy constants")
        return value

    def __call__(self, value: Any) -> Any:
        import sympy as sp

        if isinstance(value, sp.Basic):
            return value
        if isinstance(value, int | float | complex | fractions.Fraction | decimal.Decimal):
            return sp.sympify(value, strict=True)
        raise _forbid(
            "S(...) accepts only numbers and SymPy objects: for text use sp.Symbol or sp.Rational"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        raise _forbid("S cannot be modified")

    def __repr__(self) -> str:
        return "S"


def _is_singleton_registry(value: Any) -> bool:
    from sympy.core.singleton import SingletonRegistry

    return isinstance(value, SingletonRegistry)


def wrap_value(value: Any) -> Any:
    if isinstance(value, types.ModuleType):
        if not _module_allowed(value.__name__):
            raise _forbid(f"Module '{value.__name__}' not allowed")
        return ModuleProxy(value)
    if _is_singleton_registry(value):
        return SingletonProxy(value)
    return value


def import_module(name: str, line: int) -> types.ModuleType:
    if not _module_allowed(name):
        raise _forbid(
            f"import of '{name}' not allowed (allowed: {', '.join(sorted(ALLOWED_TOP))})", line
        )
    return importlib.import_module(name)


# ------------------------------------------------------------------ builtins with limits


def _guard_len(n: int) -> None:
    if n > limits.SANDBOX_SEQ_MAX:
        raise _limit(f"Sequence of {n:,} elements: maximum {limits.SANDBOX_SEQ_MAX:,}.")


def _materialize(it: Any) -> list[Any]:
    if hasattr(it, "__len__"):
        try:
            _guard_len(len(it))
        except (TypeError, OverflowError):
            raise _limit("Sequence too long.") from None
    out = list(itertools.islice(iter(it), limits.SANDBOX_SEQ_MAX + 1))
    _guard_len(len(out))
    return out


def _s_list(it: Any = ()) -> list[Any]:
    return _materialize(it)


def _s_tuple(it: Any = ()) -> tuple[Any, ...]:
    return tuple(_materialize(it))


def _s_set(it: Any = ()) -> set[Any]:
    return set(_materialize(it))


def _s_frozenset(it: Any = ()) -> frozenset[Any]:
    return frozenset(_materialize(it))


def _s_sorted(it: Any, *, key: Any = None, reverse: bool = False) -> list[Any]:
    return sorted(_materialize(it), key=key, reverse=reverse)


def _s_dict(*args: Any, **kwargs: Any) -> dict[Any, Any]:
    if args:
        src = args[0]
        if not isinstance(src, dict):
            src = _materialize(src)
        return dict(src, **kwargs)
    return dict(**kwargs)


def _s_bytes(*args: Any) -> bytes:
    if args and isinstance(args[0], int):
        _guard_len(args[0])
    return bytes(*args)


def _int_like(v: Any) -> int | None:
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    try:
        import sympy as sp

        if isinstance(v, sp.Integer):
            return int(v)
    except ImportError:  # pragma: no cover
        pass
    return None


def guarded_pow(a: Any, b: Any, mod: Any = None) -> Any:
    if mod is not None:
        return pow(a, b, mod)
    ia, ib = _int_like(a), _int_like(b)
    if ib is not None and ia is not None and abs(ia) > 1:
        if abs(ib) > limits.EXPONENT_MAX:
            raise ToolError(
                "LIMIT_EXCEEDED",
                f"Exponent {ib} beyond the limit of {limits.EXPONENT_MAX}.",
                "pow(a, b, m) for modular powers",
            )
        if ib > 0 and ia.bit_length() * ib > limits.INT_BITS_MAX:
            raise _limit("Integer result too large (beyond 4,000,000 bits).")
    return a**b


def guarded_mul(a: Any, b: Any) -> Any:
    ia, ib = _int_like(a), _int_like(b)
    if (
        ia is not None
        and ib is not None
        and ia.bit_length() + ib.bit_length() > limits.INT_BITS_MAX
    ):
        raise _limit("Integer result too large (beyond 4,000,000 bits).")
    for seq, k in ((a, ib), (b, ia)):
        if k is not None and isinstance(seq, str | bytes | list | tuple):
            _guard_len(len(seq) * max(k, 0))
    return a * b


def guarded_lshift(a: Any, b: Any) -> Any:
    ib = _int_like(b)
    if ib is not None and ib > limits.INT_BITS_MAX:
        raise _limit("Bit shift beyond 4,000,000.")
    return a << b


class _Output(io.TextIOBase):
    """stdout captured with a character cap."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.size = 0
        self.truncated = False

    def write(self, s: str) -> int:
        room = limits.SANDBOX_OUTPUT_MAX - self.size
        if room <= 0:
            self.truncated = True
            return len(s)
        if len(s) > room:
            self.truncated = True
            s = s[:room]
        self.parts.append(s)
        self.size += len(s)
        return len(s)

    def text(self) -> str:
        return "".join(self.parts)


# ------------------------------------------------------------------ user functions


class SandboxFunction:
    """Function defined in the sandbox (def or lambda)."""

    def __init__(
        self,
        interp: Interpreter,
        node: Any,
        scope: _Scope,
        defaults: list[Any],
        kw_defaults: list[Any],
        name: str,
    ) -> None:
        self._interp = interp
        self._node = node
        self._scope = scope
        self._defaults = defaults
        self._kw_defaults = kw_defaults
        self.__name__ = name
        self.__qualname__ = name

    def __repr__(self) -> str:
        return f"<function {self.__name__}>"

    def _bind(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
        a: ast.arguments = self._node.args
        positional = [p.arg for p in a.posonlyargs + a.args]
        local: dict[str, Any] = {}
        if len(args) > len(positional) and a.vararg is None:
            raise TypeError(
                f"{self.__name__}() takes at most {len(positional)} arguments, got {len(args)}"
            )
        for name, value in zip(positional, args, strict=False):
            local[name] = value
        if a.vararg is not None:
            local[a.vararg.arg] = tuple(args[len(positional) :])
        extra: dict[str, Any] = {}
        posonly = {p.arg for p in a.posonlyargs}
        kwonly = [k.arg for k in a.kwonlyargs]
        for k, v in kwargs.items():
            if k in local or k in posonly:
                raise TypeError(f"{self.__name__}(): repeated argument '{k}'")
            if k in positional or k in kwonly:
                local[k] = v
            elif a.kwarg is not None:
                extra[k] = v
            else:
                raise TypeError(f"{self.__name__}(): unexpected argument '{k}'")
        first_default = len(positional) - len(self._defaults)
        for i, name in enumerate(positional):
            if name not in local:
                if i >= first_default:
                    local[name] = self._defaults[i - first_default]
                else:
                    raise TypeError(f"{self.__name__}(): missing argument '{name}'")
        for name, default in zip(kwonly, self._kw_defaults, strict=True):
            if name not in local:
                if default is _NO_DEFAULT:
                    raise TypeError(f"{self.__name__}(): missing argument '{name}'")
                local[name] = default
        if a.kwarg is not None:
            local[a.kwarg.arg] = extra
        return local

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        interp = self._interp
        interp.tick()
        scope = _Scope(self._bind(args, kwargs), self._scope)
        if isinstance(self._node, ast.Lambda):
            return interp.evaluate(self._node.body, scope)
        try:
            interp.exec_block(self._node.body, scope)
        except _Return as r:
            return r.value
        return None


_NO_DEFAULT = object()


class _Scope:
    def __init__(self, names: dict[str, Any], parent: _Scope | None) -> None:
        self.names = names
        self.parent = parent
        self.global_names: set[str] = set()  # declared with `global` in a function


# ------------------------------------------------------------------ static check


def _bound_names(tree: ast.AST) -> set[str]:
    """Names the program binds somewhere (assignments, arguments, def, import, except)."""
    out: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store | ast.Del):
            out.add(n.id)
        elif isinstance(n, ast.arg):
            out.add(n.arg)
        elif isinstance(n, ast.FunctionDef | ast.ExceptHandler) and n.name:
            out.add(n.name)
        elif isinstance(n, ast.Import | ast.ImportFrom):
            out.update(a.asname or a.name.split(".")[0] for a in n.names)
    return out


def static_check(tree: ast.AST) -> None:
    bound = _bound_names(tree)
    for node in ast.walk(tree):
        kind = type(node).__name__
        line = getattr(node, "lineno", None)
        if kind in _FORBIDDEN_NODES:
            raise _forbid(f"Construct '{_FORBIDDEN_NODES[kind]}' not allowed", line)
        if isinstance(node, ast.Name) and (node.id.startswith("__") or node.id in FORBIDDEN_NAMES):
            raise _forbid(f"Name '{node.id}' not allowed", line)
        if isinstance(node, ast.Name) and node.id in BUILTIN_ONLY_NAMES and node.id not in bound:
            raise _forbid(f"{node.id}() not allowed (as a variable name it is fine)", line)
        if isinstance(node, ast.Attribute):
            _check_attr(node.attr, line)
        if isinstance(node, ast.arg) and node.arg.startswith("__"):
            raise _forbid(f"Name '{node.arg}' not allowed", line)
        if isinstance(node, ast.keyword) and node.arg is not None and node.arg.startswith("__"):
            raise _forbid(f"Name '{node.arg}' not allowed", line)
        if isinstance(node, ast.FunctionDef) and node.name.startswith("__"):
            raise _forbid(f"Name '{node.name}' not allowed", line)
        if isinstance(node, ast.ImportFrom):
            if node.level:
                raise _forbid("Relative import not allowed", line)
            for a in node.names:
                if a.name != "*":
                    _check_attr(a.name, line)
        if isinstance(node, ast.Import | ast.ImportFrom):
            for a in node.names:
                if a.asname and a.asname.startswith("__"):
                    raise _forbid(f"Name '{a.asname}' not allowed", line)
        if isinstance(node, ast.Global):
            for name in node.names:
                if name.startswith("__") or name in FORBIDDEN_NAMES:
                    raise _forbid(f"Name '{name}' not allowed", line)
        if isinstance(node, ast.ExceptHandler) and node.name and node.name.startswith("__"):
            raise _forbid(f"Name '{node.name}' not allowed", line)


# ------------------------------------------------------------------ interpreter


class Interpreter:
    def __init__(self, deadline: float, out: _Output) -> None:
        self.deadline = deadline
        self.out = out
        self.ticks = 0
        self.builtins = self._builtins()
        self.globals = _Scope({}, None)

    def _print(self, *args: Any, sep: str | None = " ", end: str | None = "\n") -> None:
        sep = " " if sep is None else sep
        end = "\n" if end is None else end
        self.out.write(sep.join(str(a) for a in args) + end)

    def _builtins(self) -> dict[str, Any]:
        b: dict[str, Any] = {
            name: __builtins__[name]
            if isinstance(__builtins__, dict)
            else getattr(__builtins__, name)
            for name in (
                "abs",
                "all",
                "any",
                "ascii",
                "bin",
                "bool",
                "callable",
                "chr",
                "complex",
                "divmod",
                "enumerate",
                "filter",
                "float",
                "hash",
                "hex",
                "int",
                "isinstance",
                "issubclass",
                "iter",
                "len",
                "map",
                "max",
                "min",
                "next",
                "oct",
                "ord",
                "range",
                "repr",
                "reversed",
                "round",
                "slice",
                "str",
                "sum",
                "zip",
                "Exception",
                "ArithmeticError",
                "ZeroDivisionError",
                "ValueError",
                "TypeError",
                "KeyError",
                "IndexError",
                "LookupError",
                "StopIteration",
                "OverflowError",
                "AssertionError",
                "NameError",
                "AttributeError",
                "RuntimeError",
                "NotImplementedError",
            )
        }
        b.update(
            print=self._print,
            pow=guarded_pow,
            list=_s_list,
            tuple=_s_tuple,
            set=_s_set,
            frozenset=_s_frozenset,
            sorted=_s_sorted,
            dict=_s_dict,
            bytes=_s_bytes,
        )
        return b

    # -- time

    def tick(self) -> None:
        self.ticks += 1
        if self.ticks & 0x3FF == 0 and time.monotonic() > self.deadline:
            raise ToolError("TIMEOUT", f"Program beyond {limits.TIMEOUT_S:.0f} s.")

    # -- names

    def lookup(self, name: str, scope: _Scope, line: int | None) -> Any:
        s: _Scope | None = scope
        while s is not None:
            if name in s.global_names:
                s = self.globals
            if name in s.names:
                return s.names[name]
            s = s.parent
        if name in self.builtins:
            return self.builtins[name]
        raise NameError(f"name '{name}' is not defined")

    def bind(self, name: str, value: Any, scope: _Scope) -> None:
        """Binds a name: in the program globals if the function declared it `global`."""
        (self.globals if name in scope.global_names else scope).names[name] = value

    def s_Global(self, node: ast.Global, scope: _Scope) -> None:
        if scope is not self.globals:
            scope.global_names.update(node.names)

    def assign(self, target: ast.AST, value: Any, scope: _Scope) -> None:
        if isinstance(target, ast.Name):
            self.bind(target.id, value, scope)
        elif isinstance(target, ast.Tuple | ast.List):
            self._unpack(target.elts, value, scope)
        elif isinstance(target, ast.Subscript):
            obj = self.evaluate(target.value, scope)
            obj[self._index(target.slice, scope)] = value
        elif isinstance(target, ast.Attribute):
            obj = self.evaluate(target.value, scope)
            self._setattr(obj, target.attr, value, target.lineno)
        elif isinstance(target, ast.Starred):
            raise SyntaxError("starred assignment target must be in a list or tuple")
        else:
            raise _forbid(
                f"Assignment to {type(target).__name__} not allowed",
                getattr(target, "lineno", None),
            )

    def _unpack(self, elts: list[ast.expr], value: Any, scope: _Scope) -> None:
        items = _materialize(value)
        star = [i for i, e in enumerate(elts) if isinstance(e, ast.Starred)]
        if not star:
            if len(items) != len(elts):
                raise ValueError(f"expected {len(elts)} values to unpack, got {len(items)}")
            for t, v in zip(elts, items, strict=True):
                self.assign(t, v, scope)
            return
        k = star[0]
        after = len(elts) - k - 1
        if len(items) < len(elts) - 1:
            raise ValueError(f"expected at least {len(elts) - 1} values, got {len(items)}")
        for t, v in zip(elts[:k], items[:k], strict=True):
            self.assign(t, v, scope)
        starred = elts[k]
        assert isinstance(starred, ast.Starred)
        self.assign(starred.value, list(items[k : len(items) - after]), scope)
        for t, v in zip(elts[k + 1 :], items[len(items) - after :], strict=True):
            self.assign(t, v, scope)

    def _setattr(self, obj: Any, name: str, value: Any, line: int | None) -> None:
        _check_attr(name, line)
        import mpmath

        allowed = (
            (isinstance(obj, type(mpmath.mp)) and name in {"dps", "prec", "pretty"})
            or (isinstance(obj, decimal.Context) and name in {"prec", "rounding", "Emin", "Emax"})
            or isinstance(obj, SandboxFunction)
        )
        if not allowed:
            raise _forbid(f"Assignment to the attribute '{name}' not allowed", line)
        setattr(obj, name, value)

    def getattr(self, obj: Any, name: str, line: int | None) -> Any:
        _check_attr(name, line)
        if isinstance(obj, ModuleProxy):
            return obj._get(name)
        return wrap_value(getattr(obj, name))

    # -- statements

    def exec_block(self, body: list[ast.stmt], scope: _Scope) -> None:
        for stmt in body:
            self.tick()
            try:
                self.run_stmt(stmt, scope)
            except _Control:
                raise
            except BaseException as exc:
                if getattr(exc, "_oce_line", None) is None:
                    with contextlib.suppress(AttributeError, TypeError):
                        exc._oce_line = stmt.lineno  # type: ignore[attr-defined]
                raise

    def run_stmt(self, node: ast.stmt, scope: _Scope) -> None:
        fn = getattr(self, "s_" + type(node).__name__, None)
        if fn is None:
            raise _forbid(
                f"Statement '{type(node).__name__}' not supported", getattr(node, "lineno", None)
            )
        fn(node, scope)

    def s_Expr(self, node: ast.Expr, scope: _Scope) -> None:
        self.evaluate(node.value, scope)

    def s_Assign(self, node: ast.Assign, scope: _Scope) -> None:
        value = self.evaluate(node.value, scope)
        for t in node.targets:
            self.assign(t, value, scope)

    def s_AnnAssign(self, node: ast.AnnAssign, scope: _Scope) -> None:
        if node.value is not None:
            self.assign(node.target, self.evaluate(node.value, scope), scope)

    def s_AugAssign(self, node: ast.AugAssign, scope: _Scope) -> None:
        t = node.target
        if isinstance(t, ast.Name):
            cur = self.lookup(t.id, scope, node.lineno)
            self.bind(t.id, self.binop(node.op, cur, self.evaluate(node.value, scope)), scope)
        elif isinstance(t, ast.Subscript):
            obj = self.evaluate(t.value, scope)
            key = self._index(t.slice, scope)
            obj[key] = self.binop(node.op, obj[key], self.evaluate(node.value, scope))
        elif isinstance(t, ast.Attribute):
            obj = self.evaluate(t.value, scope)
            cur = self.getattr(obj, t.attr, node.lineno)
            self._setattr(
                obj, t.attr, self.binop(node.op, cur, self.evaluate(node.value, scope)), node.lineno
            )
        else:
            raise _forbid("Augmented assignment not allowed", node.lineno)

    def s_Pass(self, node: ast.Pass, scope: _Scope) -> None:
        return None

    def s_Break(self, node: ast.Break, scope: _Scope) -> None:
        raise _Break

    def s_Continue(self, node: ast.Continue, scope: _Scope) -> None:
        raise _Continue

    def s_Return(self, node: ast.Return, scope: _Scope) -> None:
        raise _Return(None if node.value is None else self.evaluate(node.value, scope))

    def s_If(self, node: ast.If, scope: _Scope) -> None:
        self.exec_block(node.body if self.evaluate(node.test, scope) else node.orelse, scope)

    def s_While(self, node: ast.While, scope: _Scope) -> None:
        while self.evaluate(node.test, scope):
            self.tick()
            try:
                self.exec_block(node.body, scope)
            except _Break:
                return
            except _Continue:
                continue
        self.exec_block(node.orelse, scope)

    def s_For(self, node: ast.For, scope: _Scope) -> None:
        for item in self.evaluate(node.iter, scope):
            self.tick()
            self.assign(node.target, item, scope)
            try:
                self.exec_block(node.body, scope)
            except _Break:
                return
            except _Continue:
                continue
        self.exec_block(node.orelse, scope)

    def s_FunctionDef(self, node: ast.FunctionDef, scope: _Scope) -> None:
        defaults = [self.evaluate(d, scope) for d in node.args.defaults]
        kw_defaults = [
            _NO_DEFAULT if d is None else self.evaluate(d, scope) for d in node.args.kw_defaults
        ]
        fn: Any = SandboxFunction(self, node, scope, defaults, kw_defaults, node.name)
        for dec in reversed(node.decorator_list):
            fn = self.evaluate(dec, scope)(fn)
        self.bind(node.name, fn, scope)

    def s_Import(self, node: ast.Import, scope: _Scope) -> None:
        for alias in node.names:
            mod = import_module(alias.name, node.lineno)
            if alias.asname:
                scope.names[alias.asname] = ModuleProxy(mod)
            else:
                top = alias.name.split(".")[0]
                scope.names[top] = ModuleProxy(importlib.import_module(top))

    def s_ImportFrom(self, node: ast.ImportFrom, scope: _Scope) -> None:
        mod = ModuleProxy(import_module(node.module or "", node.lineno))
        for alias in node.names:
            if alias.name == "*":
                self._import_star(mod, scope)
                continue
            scope.names[alias.asname or alias.name] = mod._get(alias.name)

    def _import_star(self, mod: ModuleProxy, scope: _Scope) -> None:
        """`from m import *` (D38): every public name goes through the same checks as `from m import n`."""
        raw = object.__getattribute__(mod, "_m")
        names = getattr(raw, "__all__", None) or [n for n in dir(raw) if not n.startswith("_")]
        for name in names:
            if not isinstance(name, str) or name.startswith("_") or name in FORBIDDEN_NAMES:
                continue
            try:
                scope.names[name] = mod._get(name)
            except (ToolError, AttributeError, ImportError):
                continue

    def s_Delete(self, node: ast.Delete, scope: _Scope) -> None:
        for t in node.targets:
            if isinstance(t, ast.Name):
                if t.id not in scope.names:
                    raise NameError(f"name '{t.id}' is not defined")
                del scope.names[t.id]
            elif isinstance(t, ast.Subscript):
                obj = self.evaluate(t.value, scope)
                del obj[self._index(t.slice, scope)]
            else:
                raise _forbid("del allowed only on names and subscripts", node.lineno)

    def s_Assert(self, node: ast.Assert, scope: _Scope) -> None:
        if not self.evaluate(node.test, scope):
            msg = self.evaluate(node.msg, scope) if node.msg is not None else ""
            raise AssertionError(msg)

    def s_Raise(self, node: ast.Raise, scope: _Scope) -> None:
        if node.exc is None:
            raise RuntimeError("No active exception to reraise")
        exc = self.evaluate(node.exc, scope)
        if isinstance(exc, type) and issubclass(exc, Exception) and not issubclass(exc, ToolError):
            raise exc()
        if isinstance(exc, Exception) and not isinstance(exc, ToolError):
            raise exc
        raise TypeError("exceptions must derive from BaseException")

    def s_Try(self, node: ast.Try, scope: _Scope) -> None:
        try:
            try:
                self.exec_block(node.body, scope)
            except _UNCATCHABLE:
                raise
            except Exception as exc:
                for h in node.handlers:
                    kinds = Exception if h.type is None else self.evaluate(h.type, scope)
                    if isinstance(exc, kinds):
                        if h.name:
                            scope.names[h.name] = exc
                        self.exec_block(h.body, scope)
                        break
                else:
                    raise
            else:
                self.exec_block(node.orelse, scope)
        finally:
            self.exec_block(node.finalbody, scope)

    # -- expressions

    def evaluate(self, node: ast.expr, scope: _Scope) -> Any:
        fn = getattr(self, "e_" + type(node).__name__, None)
        if fn is None:
            raise _forbid(
                f"Expression '{type(node).__name__}' not supported", getattr(node, "lineno", None)
            )
        return fn(node, scope)

    def e_Constant(self, node: ast.Constant, scope: _Scope) -> Any:
        return node.value

    def e_Name(self, node: ast.Name, scope: _Scope) -> Any:
        return self.lookup(node.id, scope, node.lineno)

    def e_NamedExpr(self, node: ast.NamedExpr, scope: _Scope) -> Any:
        value = self.evaluate(node.value, scope)
        scope.names[node.target.id] = value
        return value

    def binop(self, op: ast.operator, a: Any, b: Any) -> Any:
        if isinstance(op, ast.Pow):
            return guarded_pow(a, b)
        if isinstance(op, ast.Mult):
            return guarded_mul(a, b)
        if isinstance(op, ast.LShift):
            return guarded_lshift(a, b)
        return _BIN[type(op)](a, b)

    def e_BinOp(self, node: ast.BinOp, scope: _Scope) -> Any:
        return self.binop(
            node.op, self.evaluate(node.left, scope), self.evaluate(node.right, scope)
        )

    def e_UnaryOp(self, node: ast.UnaryOp, scope: _Scope) -> Any:
        return _UNARY[type(node.op)](self.evaluate(node.operand, scope))

    def e_BoolOp(self, node: ast.BoolOp, scope: _Scope) -> Any:
        value: Any = None
        for v in node.values:
            value = self.evaluate(v, scope)
            if isinstance(node.op, ast.And) and not value:
                return value
            if isinstance(node.op, ast.Or) and value:
                return value
        return value

    def e_Compare(self, node: ast.Compare, scope: _Scope) -> Any:
        # As in Python: `a < b < c` means `(a < b) and (b < c)`; the last comparison is returned as is,
        # so `x**2 > 4` stays a SymPy relation and does not become true/false.
        left = self.evaluate(node.left, scope)
        last = len(node.ops) - 1
        res: Any = True
        for i, (op, comp) in enumerate(zip(node.ops, node.comparators, strict=True)):
            right = self.evaluate(comp, scope)
            res = _CMP[type(op)](left, right)
            if i < last and not res:
                return res
            left = right
        return res

    def e_IfExp(self, node: ast.IfExp, scope: _Scope) -> Any:
        return (
            self.evaluate(node.body, scope)
            if self.evaluate(node.test, scope)
            else self.evaluate(node.orelse, scope)
        )

    def e_Attribute(self, node: ast.Attribute, scope: _Scope) -> Any:
        return self.getattr(self.evaluate(node.value, scope), node.attr, node.lineno)

    def _index(self, node: ast.expr, scope: _Scope) -> Any:
        return self.evaluate(node, scope)

    def e_Slice(self, node: ast.Slice, scope: _Scope) -> Any:
        def ev(n: ast.expr | None) -> Any:
            return None if n is None else self.evaluate(n, scope)

        return slice(ev(node.lower), ev(node.upper), ev(node.step))

    def e_Subscript(self, node: ast.Subscript, scope: _Scope) -> Any:
        return self.evaluate(node.value, scope)[self._index(node.slice, scope)]

    def _elements(self, elts: list[ast.expr], scope: _Scope) -> list[Any]:
        out: list[Any] = []
        for e in elts:
            if isinstance(e, ast.Starred):
                out.extend(_materialize(self.evaluate(e.value, scope)))
            else:
                out.append(self.evaluate(e, scope))
            _guard_len(len(out))
        return out

    def e_List(self, node: ast.List, scope: _Scope) -> Any:
        return self._elements(node.elts, scope)

    def e_Tuple(self, node: ast.Tuple, scope: _Scope) -> Any:
        return tuple(self._elements(node.elts, scope))

    def e_Set(self, node: ast.Set, scope: _Scope) -> Any:
        return set(self._elements(node.elts, scope))

    def e_Dict(self, node: ast.Dict, scope: _Scope) -> Any:
        out: dict[Any, Any] = {}
        for k, v in zip(node.keys, node.values, strict=True):
            if k is None:
                out.update(self.evaluate(v, scope))
            else:
                out[self.evaluate(k, scope)] = self.evaluate(v, scope)
        return out

    def e_Call(self, node: ast.Call, scope: _Scope) -> Any:
        func = self.evaluate(node.func, scope)
        args = self._elements(node.args, scope)
        kwargs: dict[str, Any] = {}
        for kw in node.keywords:
            if kw.arg is None:
                extra = self.evaluate(kw.value, scope)
                for key in extra:
                    if not isinstance(key, str) or key.startswith("__"):
                        raise _forbid("Argument name not allowed", node.lineno)
                kwargs.update(extra)
            else:
                kwargs[kw.arg] = self.evaluate(kw.value, scope)
        if not callable(func):
            raise TypeError(f"'{type(func).__name__}' object is not callable")
        return func(*args, **kwargs)

    def e_Lambda(self, node: ast.Lambda, scope: _Scope) -> Any:
        defaults = [self.evaluate(d, scope) for d in node.args.defaults]
        kw_defaults = [
            _NO_DEFAULT if d is None else self.evaluate(d, scope) for d in node.args.kw_defaults
        ]
        return SandboxFunction(self, node, scope, defaults, kw_defaults, "<lambda>")

    def e_JoinedStr(self, node: ast.JoinedStr, scope: _Scope) -> Any:
        parts = []
        for v in node.values:
            parts.append(v.value if isinstance(v, ast.Constant) else self.evaluate(v, scope))
        return "".join(str(p) for p in parts)

    def e_FormattedValue(self, node: ast.FormattedValue, scope: _Scope) -> Any:
        value = self.evaluate(node.value, scope)
        if node.conversion == ord("r"):
            value = repr(value)
        elif node.conversion == ord("s"):
            value = str(value)
        elif node.conversion == ord("a"):
            value = ascii(value)
        spec = "" if node.format_spec is None else self.evaluate(node.format_spec, scope)
        return format(value, spec)

    # -- comprehension

    def _comp(self, gens: list[ast.comprehension], scope: _Scope) -> Iterator[_Scope]:
        def rec(i: int, sc: _Scope) -> Iterator[_Scope]:
            if i == len(gens):
                yield sc
                return
            g = gens[i]
            if g.is_async:
                raise _forbid("async not allowed", None)
            for item in self.evaluate(g.iter, sc):
                self.tick()
                inner = _Scope({}, sc) if i == 0 else sc
                self.assign(g.target, item, inner)
                if all(self.evaluate(c, inner) for c in g.ifs):
                    yield from rec(i + 1, inner)

        return rec(0, scope)

    def e_ListComp(self, node: ast.ListComp, scope: _Scope) -> Any:
        out = []
        for sc in self._comp(node.generators, scope):
            out.append(self.evaluate(node.elt, sc))
            if len(out) > limits.SANDBOX_SEQ_MAX:
                _guard_len(len(out))
        return out

    def e_SetComp(self, node: ast.SetComp, scope: _Scope) -> Any:
        out = set()
        for sc in self._comp(node.generators, scope):
            out.add(self.evaluate(node.elt, sc))
            if len(out) > limits.SANDBOX_SEQ_MAX:
                _guard_len(len(out))
        return out

    def e_DictComp(self, node: ast.DictComp, scope: _Scope) -> Any:
        out = {}
        for sc in self._comp(node.generators, scope):
            out[self.evaluate(node.key, sc)] = self.evaluate(node.value, sc)
            if len(out) > limits.SANDBOX_SEQ_MAX:
                _guard_len(len(out))
        return out

    def e_GeneratorExp(self, node: ast.GeneratorExp, scope: _Scope) -> Any:
        def gen() -> Iterator[Any]:
            for sc in self._comp(node.generators, scope):
                yield self.evaluate(node.elt, sc)

        return gen()


# ------------------------------------------------------------------ tool entry point


def _reset_state() -> None:
    import mpmath
    import numpy as np

    random.seed(0)
    np.random.seed(0)
    mpmath.mp.dps = 15
    mpmath.mp.pretty = False
    decimal.setcontext(decimal.Context())


def run_code(code: str) -> dict[str, Any]:
    harden()
    if len(code) > limits.SANDBOX_CODE_MAX:
        raise ToolError(
            "LIMIT_EXCEEDED", f"Code of {len(code)} characters: maximum {limits.SANDBOX_CODE_MAX}."
        )
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise ToolError(
            "PARSE_ERROR", f"SyntaxError: {exc.msg} (line {exc.lineno}).", "x = 6\nprint(x * 7)"
        ) from None
    except (ValueError, MemoryError, RecursionError) as exc:
        raise ToolError("PARSE_ERROR", f"Unreadable code: {type(exc).__name__}.") from None
    static_check(tree)
    _reset_state()
    out = _Output()
    t0 = time.monotonic()
    interp = Interpreter(t0 + max(0.5, limits.TIMEOUT_S - 0.5), out)
    body = tree.body
    last = body[-1] if body and isinstance(body[-1], ast.Expr) else None
    value: Any = None
    try:
        with contextlib.redirect_stdout(out):
            interp.exec_block(body[:-1] if last is not None else body, interp.globals)
            if last is not None:
                interp.tick()
                try:
                    value = interp.evaluate(last.value, interp.globals)
                except _Control:
                    raise
                except BaseException as exc:
                    if getattr(exc, "_oce_line", None) is None:
                        with contextlib.suppress(AttributeError, TypeError):
                            exc._oce_line = last.lineno  # type: ignore[attr-defined]
                    raise
    except (_Break, _Continue):
        raise ToolError("PARSE_ERROR", "'break' or 'continue' outside a loop.") from None
    except _Return:
        raise ToolError("PARSE_ERROR", "'return' outside a function.") from None
    except ToolError as exc:
        line = getattr(exc, "_oce_line", None)
        if line and "(line " not in exc.message:
            exc.message = f"{exc.message.rstrip('.')} (line {line})."
        raise
    except (RecursionError, MemoryError):
        raise
    except Exception as exc:
        line = getattr(exc, "_oce_line", None)
        where = f" (line {line})" if line else ""
        raise ToolError("MATH_ERROR", f"{type(exc).__name__}: {str(exc)[:300]}{where}.") from None
    text = out.text()
    value_text: str | None = None
    if value is not None:
        value_text = repr(value)
        if len(value_text) > limits.SANDBOX_OUTPUT_MAX:
            value_text = value_text[: limits.SANDBOX_OUTPUT_MAX]
            out.truncated = True
    return {
        "stdout": text,
        "value": value_text,
        "truncated": out.truncated,
        "elapsed_s": round(time.monotonic() - t0, 4),
    }


def handle(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    res = run_code(args["code"])
    if res["truncated"]:
        ctx.warn(f"Output cut at {limits.SANDBOX_OUTPUT_MAX} characters.")
    ctx.formatted = (
        res["value"] if res["value"] is not None else (res["stdout"][-200:] or "(no output)")
    )
    return res
