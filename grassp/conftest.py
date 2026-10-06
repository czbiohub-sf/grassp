"""Shared pytest fixtures for the grassp test suite."""

import contextlib
import sys

import pytest


@contextlib.contextmanager
def _genuine_module(name: str):
    """Temporarily evict ``name`` and its submodules from ``sys.modules`` and re-import it.

    Restores the previous entries on exit, whatever they were.
    """
    previous = {k: v for k, v in sys.modules.items() if k == name or k.startswith(name + ".")}
    for key in previous:
        del sys.modules[key]
    try:
        pytest.importorskip(name)
        yield
    finally:
        for key in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
            del sys.modules[key]
        sys.modules.update(previous)


@pytest.fixture(scope="session")
def real_rdata():
    """Run the test body with the genuine ``rdata`` package, not the mock ``test_io.py`` installs.

    ``grassp/tests/test_io.py`` puts a ``MagicMock`` in ``sys.modules["rdata"]`` at *import*
    time so its ``@patch("rdata.parser.parse_file")`` tests work without the optional dependency.
    That is process-wide and happens at collection, and :func:`grassp.io.read_prolocdata`
    imports ``rdata`` lazily -- so a test that reads a real ``.rda`` file would otherwise be
    parsing it with a mock in a full run while passing when its file runs alone. Skips when
    ``rdata`` is not installed.

    Session-scoped so module-scoped data fixtures can depend on it; it holds no state. Use it
    as a context manager around the read, not around the assertions::

        @pytest.fixture(scope="module")
        def adata(real_rdata):
            with real_rdata():
                return read.read_prolocdata(path)
    """
    return lambda: _genuine_module("rdata")
