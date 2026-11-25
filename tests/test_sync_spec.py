"""--sync-spec: optional mirrors come only from environment variables (no private paths in code)."""

from __future__ import annotations

import pytest

from oce_math_mcp import sync_spec

SPEC = "# Spec\n\nSpec version: **0.4.0**\n\nbody\n"


def test_mirrors_unset_means_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OCE_MATH_DATASET_DIR", raising=False)
    monkeypatch.setenv("OCE_MATH_RAG_DOC", "  ")
    assert sync_spec.env_path("OCE_MATH_DATASET_DIR") is None
    assert sync_spec.env_path("OCE_MATH_RAG_DOC") is None


def test_spec_version_read_from_markdown() -> None:
    assert sync_spec.spec_version(SPEC) == "0.4.0"


def test_new_document_gets_minimal_frontmatter() -> None:
    doc = sync_spec.rag_doc(None, SPEC, "0.4.0", "first")
    assert doc.startswith("---\nversion: 0.4.0\n---\n\nDocument changelog:\n- 0.4.0 - ")
    assert doc.rstrip("\n").endswith("body")


def test_existing_frontmatter_kept_and_version_bumped() -> None:
    old = (
        "---\nkb: X\nrelease: 0.3.0\ntags: [a, b]\n---\n\nDocument changelog:\n"
        "- 0.3.0 - 2026-11-20 - old note\n\n# Spec\n\nSpec version: **0.3.0**\n\nold body"
        "\n\n---\n\n# Appendix\n\nkept text\n"
    )
    doc = sync_spec.rag_doc(old, SPEC, "0.4.0", "new note")
    assert doc.startswith("---\nkb: X\nrelease: 0.4.0\ntags: [a, b]\n---\n")
    assert "- 0.4.0 - " in doc and "new note" in doc
    assert "- 0.3.0 - 2026-11-20 - old note" in doc
    assert "old body" not in doc and "\nbody" in doc
    assert doc.endswith("# Appendix\n\nkept text\n")


def test_same_version_not_logged_twice() -> None:
    first = sync_spec.rag_doc(None, SPEC, "0.4.0", "note")
    again = sync_spec.rag_doc(first, SPEC, "0.4.0", "note")
    assert again.count("- 0.4.0 - ") == 1
