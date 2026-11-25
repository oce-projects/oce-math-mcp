"""README header: donation buttons first, then only true badges; the version badge follows VERSION."""

from __future__ import annotations

import re

from tests.conftest import ROOT

README = (ROOT / "README.md").read_text(encoding="utf-8")
VERSION = (ROOT / "VERSION").read_text(encoding="utf-8").strip()


def test_version_badge_matches_version_file() -> None:
    m = re.search(r"img\.shields\.io/badge/version-([0-9A-Za-z.\-]+?)-blue", README)
    assert m, "version badge missing"
    assert m.group(1).replace("--", "-") == VERSION


def test_readme_opens_with_the_two_support_buttons() -> None:
    first = README.splitlines()[:2]
    assert first[0].startswith("[![Buy me a coffee]")
    assert "https://buymeacoffee.com/oscarcontatoengineering" in first[0]
    assert first[1].startswith("[![Donate via PayPal]")
    assert "https://www.paypal.biz/OSCARCONTATO94" in first[1]


def test_no_ci_badge_without_ci() -> None:
    # The project runs its tests locally, not in GitHub Actions: a CI badge would be false.
    assert "actions/workflows" not in README
    assert not (ROOT / ".github" / "workflows").exists()


def test_license_badge_matches_pyproject() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'license = "Apache-2.0"' in pyproject
    assert "badge/license-Apache--2.0-" in README
