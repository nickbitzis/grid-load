"""Shared fixtures: the full four-year dataset (under a second), a small one, and one quick pipeline run."""

from __future__ import annotations

from datetime import date

import pytest

from gridload.features import node_data
from gridload.generator import GenConfig, generate
from gridload.pipeline import PipelineConfig, build_artifacts, roundtrip


@pytest.fixture(scope="session")
def ds():
    return generate(GenConfig(seed=42))


@pytest.fixture(scope="session")
def nd(ds):
    return node_data(ds)


@pytest.fixture(scope="session")
def tiny():
    """150 days in spring, no events: for model unit tests."""
    return generate(GenConfig(seed=3, end=date(2025, 5, 31), days=150))


@pytest.fixture(scope="session")
def art():
    return roundtrip(build_artifacts(PipelineConfig.quick(), log=lambda m: None))
