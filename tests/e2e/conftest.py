"""The end-to-end preset runs are opt-in: they take a while (every preset at time scale 10, twice)."""


def pytest_configure(config):
    config.addinivalue_line("markers", "e2e: runs every scenario preset end to end (skipped unless WARDZERO_E2E=1)")
