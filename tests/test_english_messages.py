"""No Italian left in the server texts (D81, D97).

The translation check of D61 looked for words outside an English vocabulary, so Italian words that are also English
words slipped through: five error messages still read "Serve ..." ("... is needed") or "Per ... serve ...". This test
reads every string literal of the package and fails on the Italian verbs and phrases that such a check cannot see.

D97 (25/11/2026): on Python 3.12 and later the text of an f-string is an FSTRING_MIDDLE token, not a STRING token,
so the test did not read f-strings at all; twelve texts were still Italian ("non usa", "non è 4x4", "box o cylinder",
"polyroots non converge", "NON coincide"...). The test now reads both tokens, and also looks for the Italian "o"
("or") between two words, "non" before a verb, "usa", "Norma", "per zero" and a standalone "è".
"""

from __future__ import annotations

import io
import re
import tokenize
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "oce_math_mcp"
ITALIAN = re.compile(
    r"\b(Serve|Servono|Manca|Mancano|Deve|Devono|Occorre|Bisogna)\b|\bPer '|\bserve (params|matrix)"
    r"|[\w'] o [\w']|\bnon (usa|è|converge|coincide)\b|\bNON\b|\busa\b|\bNorma\b|\bper zero\b"
    r"|(?<!\w)è(?!\w)|^'coincide'$"
)
TEXT_TOKENS = {tokenize.STRING} | (
    {tokenize.FSTRING_MIDDLE} if hasattr(tokenize, "FSTRING_MIDDLE") else set()
)


def italian_texts(source: str) -> list[tuple[int, str]]:
    return [
        (tok.start[0], tok.string)
        for tok in tokenize.generate_tokens(io.StringIO(source).readline)
        if tok.type in TEXT_TOKENS and ITALIAN.search(tok.string)
    ]


def test_no_italian_in_string_literals() -> None:
    found = []
    for path in sorted(PACKAGE.rglob("*.py")):
        source = path.read_bytes().decode("utf-8")
        for line, text in italian_texts(source):
            found.append(f"{path.relative_to(PACKAGE)}:{line}: {text[:80]}")
    assert not found, "\n".join(found)


def test_the_check_reads_f_strings_and_finds_the_d97_phrases() -> None:
    for src in (
        "x = f\"shape '{s}' non usa: {w}.\"\n",
        'x = f"params.matrices[{i}] non è 4x4."\n',
        'x = "params.shape: box o cylinder."\n',
        "x = \"Norm: expr = '1', '2', 'oo' o 'fro'.\"\n",
        'x = "polyroots non converge."\n',
        'x = "Modulo per zero."\n',
        "x = f\"({'coincide' if a else 'NON coincide'})\"\n",
        'x = "events: usa both."\n',
    ):
        assert italian_texts(src), src
    for src in (
        'x = "Division by zero."\n',
        'x = "Ampère\'s force law; Bézout coefficients; Lamé."\n',
        "x = f\"'{name}' contains non-finite values.\"\n",
        'x = "bounds [lo, hi], one per variable; o(n) and e = 2."\n',
        'x = "params.shape: box or cylinder."\n',
        'x = "The two points coincide."\n',
    ):
        assert not italian_texts(src), src
