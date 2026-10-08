"""The selected Muon stack ABI must be admitted by the actual native compiler."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "tools/muon-compile.py"
SPEC = importlib.util.spec_from_file_location("muon_compile_preflight", SOURCE)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_native_abi_probe_passes_selected_stride_to_compiler():
    with patch.object(MODULE.subprocess, "run", return_value=SimpleNamespace(
        returncode=0, stderr="", stdout="")) as call:
        MODULE.probe_native_abi(Path("/compiler/clang"), 16)
    command = call.call_args.args[0]
    assert "-riscv-stack-word-stride=16" in command
    assert "+vortex" in command


def test_native_abi_probe_fails_before_lowering_on_unsupported_stride():
    with patch.object(MODULE.subprocess, "run", return_value=SimpleNamespace(
        returncode=1, stderr="unknown option", stdout="")):
        with pytest.raises(ValueError, match="stack-word-stride=16: unknown option"):
            MODULE.probe_native_abi(Path("/compiler/clang"), 16)
