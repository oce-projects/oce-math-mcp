"""No Italian left in the server texts (D81).

The translation check of D61 looked for words outside an English vocabulary, so Italian words that are also English
words slipped through: five error messages still read "Serve ..." ("... is needed") or "Per ... serve ...". This test
reads every string literal of the package and fails on the Italian verbs and phrases that such a check cannot see.
"""

from __future__ import annotations

import io
import re
import tokenize
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "oce_math_mcp"
ITALIAN = re.compile(
    r"\b(Serve|Servono|Manca|Mancano|Deve|Devono|Occorre|Bisogna)\b|\bPer '|\bserve (params|matrix)"
)


def test_no_italian_in_string_literals() -> None:
    found = []
    for path in sorted(PACKAGE.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.STRING and ITALIAN.search(tok.string):
                found.append(f"{path.relative_to(PACKAGE)}:{tok.start[0]}: {tok.string[:80]}")
    assert not found, "\n".join(found)
