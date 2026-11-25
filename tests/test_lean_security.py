"""lean_check: model code must not run programs or add axioms (D36).

Lean runs code during compilation: `#eval` of an IO action, `elab`/`macro` commands,
attributes that register elaborators (`@[command_elab]`), `run_cmd`. An `axiom` or `native_decide`
would make a false proof "valid". The filter works on the text before calling Lean,
so these tests do not need Lean installed.
"""

from __future__ import annotations

import pytest

from oce_math_mcp.dispatch import run_tool
from oce_math_mcp.tools import lean_check

FORBIDDEN_CODE = [
    '#eval IO.Process.run { cmd := "calc.exe" }',
    "#exit",
    '#leansearch "prime"',
    'run_cmd Lean.logInfo "x"',
    "theorem t : True := by run_tac pure ()",
    'elab "boom" : command => pure ()',
    "elab_rules : command | `(boom) => pure ()",
    'macro "boom" : command => `(#check 1)',
    "macro_rules | `(boom) => `(#check 1)",
    'syntax "boom" : command',
    "declare_syntax_cat boom",
    "initialize pure ()",
    "builtin_initialize pure ()",
    "unsafe def f : Nat := 1",
    "@[implemented_by g] def f : Nat := 1",
    '@[extern "c_f"] opaque f : Nat',
    "axiom cheat : False",
    "theorem t : 2 + 2 = 4 := by native_decide",
    "theorem t : 2 + 2 = 4 := by decide +native",
    "theorem t : 2 + 2 = 4 := by decide (config := { native := true })",
    "theorem t : 2 + 2 = 4 := Lean.ofReduceBool _ _ rfl",
    "@[command_elab Lean.Parser.Command.check] def evil : Lean.Elab.Command.CommandElab := fun _ => pure ()",
    "@[simp, tactic Lean.Parser.Tactic.done] def evil : Lean.Elab.Tactic.Tactic := fun _ => pure ()",
    "attribute [command_elab Lean.Parser.Command.check] evil",
    "@[init] def boot : IO Unit := pure ()",
    "simproc boom (_) := fun _ => pure .continue",
    "register_simp_attr boom",
    "set_option debug.skipKernelTC true in\ntheorem t : 1 = 2 := rfl",
    "set_option compiler.extract_closed false",
    "import Lake\n#check 1",
    "import Mathlib\nimport Lean.Elab.Command\n#check 1",
    "theorem t (p : Prop) : p := by polyrith",
    "theorem t (x : BitVec 8) : x + 0 = x := by bv_decide",
]

ALLOWED_CODE = [
    "theorem t : 2 + 2 = 4 := by norm_num",
    "import Mathlib\nexample (a b : ℕ) : a + b = b + a := by omega",
    "import Mathlib.Tactic\n#check Nat.add_comm",
    "#print axioms Nat.add_comm",
    "@[simp] theorem my_add (n : ℕ) : n + 0 = n := rfl",
    "@[simp, local ext] theorem s (n : ℕ) : n = n := rfl",
    "attribute [local simp] Nat.add_comm",
    "set_option maxHeartbeats 400000 in\ntheorem t : 2 + 2 = 4 := by norm_num",
    "set_option linter.unusedVariables false",
    "theorem macroeconomics (x : ℕ) : x = x := rfl  -- names that contain forbidden words",
    "def axiomatic_count : ℕ := 3",
    "theorem t : True := by decide",
    "noncomputable def f : ℝ → ℝ := fun x => x ^ 2",
]


@pytest.mark.parametrize("code", FORBIDDEN_CODE)
def test_forbidden_lean_code_is_rejected_before_lean(code: str) -> None:
    resp = run_tool("lean_check", {"code": code})
    assert resp["ok"] is False
    assert resp["error"]["code"] == "FORBIDDEN", (code, resp["error"])


@pytest.mark.parametrize("code", ALLOWED_CODE)
def test_allowed_lean_code_passes_the_filter(code: str) -> None:
    lean_check.check_source(code)  # no exception


def test_forbidden_message_names_the_construct() -> None:
    resp = run_tool("lean_check", {"code": "#eval 1"})
    assert "#eval" in resp["error"]["message"]
