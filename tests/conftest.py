"""Test-wide defaults. The automated tests run on the stub world: `hhumans run` without --sim would
otherwise pick Isaac Sim on a machine that has it (subprocesses inherit this too)."""
import os

os.environ.setdefault("WARDZERO_SIM", "stub")
# a test that forgets to ask for the fake model can never spend money
os.environ.setdefault("WARDZERO_TEST_FAKE_LLM", "1")
