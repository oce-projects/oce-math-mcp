"""Publishability: the files that git would publish contain no personal or private data.

Scanned: tracked files plus new files that are not ignored (`git ls-files --cached --others
--exclude-standard`). Allowed exceptions, exact text only: the two donation URLs of the README
and the copyright lines of LICENSE and NOTICE (legal attribution of the Apache-2.0 licence).
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

from tests.conftest import ROOT

THIS = "tests/test_publishable.py"

PATTERNS = {
    "private IPv4 address": re.compile(
        r"\b(?:192\.168|10\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}\b"
    ),
    "e-mail address": re.compile(r"\b[\w.+-]+@[\w-]+\.[A-Za-z]{2,}\b"),
    "GitHub token": re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}|github_pat_"),
    "API key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}|\bAKIA[0-9A-Z]{16}\b"),
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    # Other drives, or a real user folder; C:/Windows and the fake C:/Users/x/ of the tests are fine.
    # The look-ahead skips string escapes such as "class P:\n".
    "local drive path": re.compile(
        r"(?<![\w$])(?:[ABD-Z]:[\\/](?![ntr0'\"])|C:[\\/]Users[\\/](?!x[\\/]))"
    ),
    "UNC path": re.compile(r"\\\\(?!server\\)[\w.-]+\\"),
    "personal name": re.compile(r"oscar|contato|rasco94", re.IGNORECASE),
    "home network names": re.compile(
        r"openclaw|diana-rag|KB_TECNICA|_DATASET_ADDESTRAMENTO|KB_SANITA|KB_STUDI", re.IGNORECASE
    ),
}

ALLOWED = (
    "https://buymeacoffee.com/oscarcontatoengineering",
    "https://www.paypal.biz/OSCARCONTATO94",
    "Copyright 2026 Oscar Contato",
    "This product includes software developed by Oscar Contato.",
)


def _published_files() -> list[str]:
    out = subprocess.run(
        [shutil.which("git") or "git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [f for f in out.stdout.splitlines() if f and f != THIS]


def _findings() -> list[str]:
    found: list[str] = []
    for rel in _published_files():
        path = ROOT / rel
        if not path.is_file() or path.stat().st_size > 4_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary data files (.gz) hold numbers only
        for n, line in enumerate(text.splitlines(), 1):
            clean = line
            for ok in ALLOWED:
                clean = clean.replace(ok, "")
            for label, pat in PATTERNS.items():
                m = pat.search(clean)
                if m:
                    found.append(f"{rel}:{n}: {label}: {m.group(0)!r}")
    return found


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_no_private_data_in_published_files() -> None:
    found = _findings()
    assert not found, f"{len(found)} findings:\n" + "\n".join(found[:60])


def test_patterns_catch_known_cases() -> None:
    samples = {
        "private IPv4 address": "server at 192.168.1.9",
        "e-mail address": "write to someone@example.com",
        "local drive path": r"copy to W:\data",
        "personal name": "approved by Oscar",
    }
    for label, text in samples.items():
        assert PATTERNS[label].search(text), label
    assert PATTERNS["local drive path"].search(r"C:\Users\someone\file")
    assert PATTERNS["UNC path"].search(r"\\nas-box\share")
    assert not PATTERNS["local drive path"].search("C:/Users/x/lean-does-not-exist")
    assert not PATTERNS["local drive path"].search("open('C:/Windows/win.ini')")
    assert not PATTERNS["local drive path"].search(r'"class P:\n    pass"')
    assert not PATTERNS["private IPv4 address"].search("doi:10.1088/1674-1137/abddb0")
