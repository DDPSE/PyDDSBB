import pytest
import numpy as np
import random


def pytest_addoption(parser):
    parser.addoption(
        "--with-solvers",
        action="store_true",
        default=False,
        help="Run tests that require external optimization solvers",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "solver: test that requires external optimization solvers",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--with-solvers"):
        skip_solver = pytest.mark.skip(
            reason="need --with-solvers to run solver-dependent tests"
        )
        for item in items:
            if "solver" in item.keywords:
                item.add_marker(skip_solver)


@pytest.fixture(autouse=True)
def _seed_rng():
    """Make tests deterministic by seeding RNGs before each test."""
    np.random.seed(0)
    random.seed(0)
