"""Synchronises the JSON-Schema contract everywhere it must be identical.

Source: docs/spec/oce_math_mcp_tools.json. Targets:
1. src/oce_math_mcp/tools.json (same bytes);
2. the ```json``` block of docs/spec/SPEC_OCE_MATH_MCP_V3.md, between the SCHEMAS-START/END markers;
3. optional dataset folder (OCE_MATH_DATASET_DIR): .json and .md files (same bytes as the repo);
4. optional spec document (OCE_MATH_RAG_DOC): its frontmatter is kept, then the changelog, the
   full spec and the kept appendix.

Writes go through a .tmp file, then os.replace, then a byte check (safe on network drives).
An optional target is skipped when its variable is not set or its folder is not reachable.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SPEC_JSON = ROOT / "docs" / "spec" / "oce_math_mcp_tools.json"
SPEC_MD = ROOT / "docs" / "spec" / "SPEC_OCE_MATH_MCP_V3.md"
PKG_JSON = Path(__file__).resolve().parent / "tools.json"
_START, _END = "<!-- SCHEMAS-START -->", "<!-- SCHEMAS-END -->"
_APPENDIX = "\n\n---\n\n# Appendix"
_VERSION_RE = re.compile(r"Spec version: \*\*(\d+\.\d+\.\d+)\*\*")


def env_path(name: str) -> Path | None:
    """Path from an environment variable; None when the variable is unset or blank."""
    value = os.environ.get(name, "").strip()
    return Path(value) if value else None


def _write(path: Path, data: bytes) -> dict[str, Any]:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    if tmp.stat().st_size != len(data):
        raise OSError(f"{tmp}: wrong size after writing")
    os.replace(tmp, path)
    back = path.read_bytes()
    if back != data:
        raise OSError(f"{path}: different content after writing")
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()[:16]}


def spec_version(md: str) -> str:
    m = _VERSION_RE.search(md)
    if not m:
        raise ValueError("spec version not found in the markdown")
    return m.group(1)


def embed_json(md: str, json_text: str) -> str:
    i, j = md.index(_START), md.index(_END)
    return md[: i + len(_START)] + "\n```json\n" + json_text + "```\n" + md[j:]


def rag_doc(old: str | None, md: str, version: str, note: str | None) -> str:
    """Spec document for a knowledge base: frontmatter, changelog, full spec, kept appendix.

    An existing frontmatter is kept as it is: only a value equal to the previous spec version
    becomes the new version. A new document gets a minimal frontmatter.
    """
    appendix = ""
    changelog: list[str] = []
    front = f"---\nversion: {version}\n---\n"
    if old:
        k = old.find(_APPENDIX)
        appendix = old[k:] if k >= 0 else ""
        m = re.search(r"^---\n.*?\n---\n", old, flags=re.DOTALL)
        if m:
            front = m.group(0)
            prev = _VERSION_RE.search(old)
            if prev:
                pattern = rf"(?m)^([\w-]+): {re.escape(prev.group(1))}$"
                front = re.sub(pattern, lambda g: f"{g.group(1)}: {version}", front)
        c = re.search(r"Document changelog:\n(.*?)\n\n", old, flags=re.DOTALL)
        if c:
            changelog = c.group(1).split("\n")
    if not any(line.startswith(f"- {version} ") for line in changelog):
        changelog.insert(0, f"- {version} - {date.today():%Y-%m-%d} - {note or 'spec updated'}")
    body = md.rstrip("\n")
    return front + "\nDocument changelog:\n" + "\n".join(changelog) + "\n\n" + body + appendix


def sync(note: str | None = None) -> dict[str, Any]:
    raw = SPEC_JSON.read_bytes()
    text = raw.decode("utf-8")
    tools = json.loads(text)
    if len(tools) != 10 or not text.endswith("\n"):
        raise ValueError("the contract must have 10 tools and end with a newline")
    report: dict[str, Any] = {"ok": True, "written": [], "skipped": []}
    report["written"].append(_write(PKG_JSON, raw))
    md = embed_json(SPEC_MD.read_text(encoding="utf-8"), text)
    md_bytes = md.encode("utf-8")
    if SPEC_MD.read_bytes() != md_bytes:
        report["written"].append(_write(SPEC_MD, md_bytes))
    version = spec_version(md)
    report["spec_version"] = version
    dataset_dir = env_path("OCE_MATH_DATASET_DIR")
    if dataset_dir is None:
        report["skipped"].append("OCE_MATH_DATASET_DIR not set")
    elif dataset_dir.is_dir():
        report["written"].append(_write(dataset_dir / SPEC_JSON.name, raw))
        report["written"].append(_write(dataset_dir / SPEC_MD.name, md_bytes))
    else:
        report["skipped"].append(f"{dataset_dir} not reachable")
    doc_path = env_path("OCE_MATH_RAG_DOC")
    if doc_path is None:
        report["skipped"].append("OCE_MATH_RAG_DOC not set")
    elif doc_path.parent.is_dir():
        old = doc_path.read_text(encoding="utf-8") if doc_path.is_file() else None
        new = rag_doc(old, md, version, note)
        if md.rstrip("\n") not in new:
            raise ValueError("spec document: the spec is not contained in full")
        report["written"].append(_write(doc_path, new.encode("utf-8")))
    else:
        report["skipped"].append(f"{doc_path.parent} not reachable")
    return report
