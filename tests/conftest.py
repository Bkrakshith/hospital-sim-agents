"""Test-wide defaults. The automated tests run on the stub world: `hhumans run` without --sim would
otherwise pick Isaac Sim on a machine that has it (subprocesses inherit this too)."""
import os

os.environ.setdefault("WARDZERO_SIM", "stub")
