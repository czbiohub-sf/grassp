"""Guardrails on the public API surface.

The audit that prompted the annotation refactor found functions that were defined,
docstringed and unreachable, and doc pages listing names that had moved. These keep the
three views -- what is exported, what is documented, and what has a stub -- in step.
"""

import inspect
import re

from pathlib import Path

import pytest

import grassp as gr

DOCS = Path(__file__).resolve().parents[2] / "docs" / "source"
GENERATED = DOCS / "generated"

NAMESPACES = {"tl": gr.tl, "pl": gr.pl, "pp": gr.pp, "io": gr.io, "ds": gr.ds}


def _listed(page: str, prefix: str) -> set[str]:
    text = (DOCS / "api" / page).read_text()
    return set(re.findall(rf"^   {prefix}\.(\w+)$", text, re.M))


@pytest.mark.parametrize(
    "page,prefix",
    [("tools.md", "tl"), ("plotting.md", "pl"), ("preprocessing.md", "pp")],
)
class TestDocsMatchTheApi:
    def test_every_documented_name_exists(self, page, prefix):
        namespace = NAMESPACES[prefix]
        missing = sorted(n for n in _listed(page, prefix) if not hasattr(namespace, n))
        assert not missing, f"{page} lists names that do not exist: {missing}"

    def test_every_documented_name_has_a_stub(self, page, prefix):
        missing = sorted(
            n
            for n in _listed(page, prefix)
            if not (GENERATED / f"grassp.{prefix}.{n}.rst").exists()
        )
        assert not missing, f"no autosummary stub for: {missing}"


class TestToolsNamespace:
    def test_all_matches_the_documentation_exactly(self):
        """grassp.tools declares __all__, so the two can be compared outright."""
        from grassp import tools

        listed = _listed("tools.md", "tl")
        assert set(tools.__all__) == listed

    def test_all_resolves_and_holds_no_modules(self):
        from grassp import tools

        for name in tools.__all__:
            assert hasattr(tools, name), f"__all__ names {name!r}, which is absent"
            assert not inspect.ismodule(getattr(tools, name)), f"{name!r} is a module"

    def test_no_public_callable_is_unreachable(self):
        """The failure mode this refactor started from: prune_markers and
        annotation_confusion_matrix were defined, fully documented and in no namespace."""
        from grassp import tools

        unreachable = []
        for module_name in ("localization", "diffusion", "scoring", "tagm", "clustering"):
            module = getattr(tools, module_name)
            for name, obj in vars(module).items():
                if name.startswith("_") or not callable(obj):
                    continue
                if getattr(obj, "__module__", "") != module.__name__:
                    continue
                if not hasattr(gr.tl, name):
                    unreachable.append(f"{module_name}.{name}")
        assert not unreachable, f"public-named but unreachable: {sorted(unreachable)}"
