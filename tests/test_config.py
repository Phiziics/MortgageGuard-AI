from pathlib import Path

import pytest

from src.config import (
    PROJECT_ROOT,
    config,
    get_required_environment_variable,
    load_yaml_config,
    resolve_project_path,
)


def test_project_name() -> None:
    """Confirm that the central configuration loads the project name."""

    assert config.project_name == "MortgageGuard-AI"


def test_vintage_years() -> None:
    """Confirm that the expected Freddie Mac vintages are configured."""

    assert config.vintage_years == [
        2023,
        2024,
        2025,
    ]


def test_random_seed() -> None:
    """Confirm that the reproducibility seed loads correctly."""

    assert config.random_seed == 42


def test_resolve_relative_project_path() -> None:
    """Confirm that relative paths resolve from the repository root."""

    result = resolve_project_path(
        "data/raw"
    )

    assert result == PROJECT_ROOT / "data/raw"
    assert result.is_absolute()


def test_resolve_absolute_path(
    tmp_path: Path,
) -> None:
    """Confirm that absolute paths remain unchanged."""

    result = resolve_project_path(
        str(tmp_path)
    )

    assert result == tmp_path


def test_load_yaml_config(
    tmp_path: Path,
) -> None:
    """Confirm that a valid YAML file loads as a dictionary."""

    yaml_path = tmp_path / "test.yaml"

    yaml_path.write_text(
        """
project:
  name: Test Project
  random_seed: 42
""".strip(),
        encoding="utf-8",
    )

    result = load_yaml_config(
        yaml_path
    )

    assert result["project"]["name"] == "Test Project"
    assert result["project"]["random_seed"] == 42


def test_missing_yaml_file_raises_error(
    tmp_path: Path,
) -> None:
    """Confirm that a missing configuration file raises a clear error."""

    missing_path = (
        tmp_path
        / "missing.yaml"
    )

    with pytest.raises(
        FileNotFoundError,
        match="Configuration file was not found",
    ):
        load_yaml_config(
            missing_path
        )


def test_invalid_yaml_structure_raises_error(
    tmp_path: Path,
) -> None:
    """Confirm that YAML must contain a dictionary at the root."""

    yaml_path = tmp_path / "invalid.yaml"

    yaml_path.write_text(
        """
- item_one
- item_two
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="must contain a YAML dictionary",
    ):
        load_yaml_config(
            yaml_path
        )


def test_missing_required_environment_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Confirm that missing required environment variables are rejected."""

    variable_name = (
        "MORTGAGEGUARD_TEST_VARIABLE"
    )

    monkeypatch.delenv(
        variable_name,
        raising=False,
    )

    with pytest.raises(
        EnvironmentError,
        match="Required environment variable",
    ):
        get_required_environment_variable(
            variable_name
        )


def test_project_directories_exist() -> None:
    """Confirm that the configuration manager creates required folders."""

    required_directories = [
        config.paths.raw_data,
        config.paths.bronze_data,
        config.paths.silver_data,
        config.paths.gold_data,
        config.paths.artifacts,
        config.paths.models,
        config.paths.reports,
        config.paths.prediction_logs,
        config.paths.monitoring_metrics,
        config.paths.monitoring_reports,
        config.paths.monitoring_incidents,
    ]

    for directory in required_directories:
        assert directory.exists()
        assert directory.is_dir()