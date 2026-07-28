from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


# Load environment variables from the local .env file.
load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ProjectPaths:
    """Centralized paths used throughout MortgageGuard AI."""

    project_root: Path
    data_config: Path
    model_config: Path
    monitoring_config: Path

    raw_data: Path
    bronze_data: Path
    silver_data: Path
    gold_data: Path

    artifacts: Path
    models: Path
    reports: Path

    prediction_logs: Path
    monitoring_metrics: Path
    monitoring_reports: Path
    monitoring_incidents: Path


def resolve_project_path(path_value: str) -> Path:
    """Convert a relative configuration path into an absolute project path."""

    path = Path(path_value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def get_required_environment_variable(name: str) -> str:
    """Return a required environment variable or raise a clear error."""

    value = os.getenv(name)

    if value is None or not value.strip():
        raise EnvironmentError(
            f"Required environment variable '{name}' is missing."
        )

    return value.strip()


def load_yaml_config(file_path: Path) -> dict[str, Any]:
    """Load and validate a YAML configuration file."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Configuration file was not found: {file_path}"
        )

    with file_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            f"Configuration file must contain a YAML dictionary: {file_path}"
        )

    return config


def build_project_paths() -> ProjectPaths:
    """Build all project paths from environment variables."""

    return ProjectPaths(
        project_root=PROJECT_ROOT,
        data_config=resolve_project_path(
            get_required_environment_variable("DATA_CONFIG_PATH")
        ),
        model_config=resolve_project_path(
            get_required_environment_variable("MODEL_CONFIG_PATH")
        ),
        monitoring_config=resolve_project_path(
            get_required_environment_variable("MONITORING_CONFIG_PATH")
        ),
        raw_data=resolve_project_path(
            get_required_environment_variable("RAW_DATA_DIRECTORY")
        ),
        bronze_data=resolve_project_path(
            get_required_environment_variable("BRONZE_DATA_DIRECTORY")
        ),
        silver_data=resolve_project_path(
            get_required_environment_variable("SILVER_DATA_DIRECTORY")
        ),
        gold_data=resolve_project_path(
            get_required_environment_variable("GOLD_DATA_DIRECTORY")
        ),
        artifacts=resolve_project_path(
            get_required_environment_variable("ARTIFACTS_DIRECTORY")
        ),
        models=resolve_project_path(
            get_required_environment_variable("MODELS_DIRECTORY")
        ),
        reports=resolve_project_path(
            get_required_environment_variable("REPORTS_DIRECTORY")
        ),
        prediction_logs=resolve_project_path(
            get_required_environment_variable("PREDICTION_LOG_DIRECTORY")
        ),
        monitoring_metrics=resolve_project_path(
            get_required_environment_variable(
                "MONITORING_METRICS_DIRECTORY"
            )
        ),
        monitoring_reports=resolve_project_path(
            get_required_environment_variable(
                "MONITORING_REPORT_DIRECTORY"
            )
        ),
        monitoring_incidents=resolve_project_path(
            get_required_environment_variable(
                "MONITORING_INCIDENT_DIRECTORY"
            )
        ),
    )


def create_project_directories(paths: ProjectPaths) -> None:
    """Create project output directories when they do not already exist."""

    directories = [
        paths.raw_data,
        paths.bronze_data,
        paths.silver_data,
        paths.gold_data,
        paths.artifacts,
        paths.models,
        paths.reports,
        paths.prediction_logs,
        paths.monitoring_metrics,
        paths.monitoring_reports,
        paths.monitoring_incidents,
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


class ConfigManager:
    """Load and expose project configuration from one central location."""

    def __init__(self) -> None:
        self.paths = build_project_paths()

        self.data = load_yaml_config(
            self.paths.data_config
        )

        self.model = load_yaml_config(
            self.paths.model_config
        )

        self.monitoring = load_yaml_config(
            self.paths.monitoring_config
        )

        create_project_directories(
            self.paths
        )

    @property
    def project_name(self) -> str:
        """Return the configured project name."""

        return self.data["project"]["name"]

    @property
    def vintage_years(self) -> list[int]:
        """Return the Freddie Mac vintage years used by the project."""

        return self.data["source"]["vintage_years"]

    @property
    def random_seed(self) -> int:
        """Return the reproducibility seed."""

        return int(
            self.data["project"]["random_seed"]
        )

    @property
    def mlflow_tracking_uri(self) -> str:
        """Return the MLflow tracking URI."""

        return get_required_environment_variable(
            "MLFLOW_TRACKING_URI"
        )

    @property
    def mlflow_experiment_name(self) -> str:
        """Return the MLflow experiment name."""

        return get_required_environment_variable(
            "MLFLOW_EXPERIMENT_NAME"
        )

    @property
    def registered_model_name(self) -> str:
        """Return the MLflow registered model name."""

        return get_required_environment_variable(
            "MLFLOW_REGISTERED_MODEL_NAME"
        )


config = ConfigManager()