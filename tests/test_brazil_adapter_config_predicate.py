from __future__ import annotations

import pytest

import run_pipeline
from src.pipeline_runtime import config_runtime


@pytest.mark.parametrize(
    ("adapter_config", "expected"),
    [
        ({}, False),
        ({"load_data": "src.data_preprocessing.load_data.run"}, False),
        ({"load_data": "projects.brazil_chik.data_adapter.load_data"}, False),
        ({"label_outbreaks": "projects.brazil_chik.label_adapter.label_outbreaks"}, True),
        (
            {
                "load_data": "src.data_preprocessing.load_data.run",
                "label_outbreaks": "projects.brazil_chik.label_adapter.label_outbreaks",
                "build_feature_matrix": "projects.brazil_chik.feature_adapter.build_feature_matrix",
            },
            True,
        ),
        (
            {
                "load_data": "projects.brazil_chik.data_adapter.load_data",
                "label_outbreaks": "projects.brazil_chik.label_adapter.label_outbreaks",
            },
            True,
        ),
        (
            {
                "load_data": "third_party.adapters.load_data",
                "label_outbreaks": "projects.brazil_chik.label_adapter.label_outbreaks",
            },
            False,
        ),
        (
            {
                "load_data": "src.data_preprocessing.load_data.run",
                "label_outbreaks": "src.data_preprocessing.label_outbreaks.label_outbreaks",
            },
            False,
        ),
    ],
)
def test_brazil_adapter_config_active_predicate(adapter_config: dict[str, str], expected: bool) -> None:
    assert config_runtime.is_brazil_adapter_config_active(adapter_config) is expected
    assert run_pipeline._is_brazil_adapter_config_active(adapter_config) is expected
