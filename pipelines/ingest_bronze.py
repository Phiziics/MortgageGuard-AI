from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data.bronze import (
    ingest_zip_archive,
    load_official_headers,
)


RAW_DIRECTORY = Path(
    "data/raw"
)

BRONZE_DIRECTORY = Path(
    "data/bronze"
)

HEADER_DIRECTORY = Path(
    "data/reference/file_headers_july_2026"
)

MANIFEST_PATH = Path(
    "reports/bronze_ingestion_manifest.csv"
)

EXPECTED_YEARS = {
    2023,
    2024,
    2025
}


def validate_manifest(
    manifest: pd.DataFrame,
) -> None:
    """Validate Bronze ingestion coverage."""

    expected_types = {
        "origination",
        "performance",
    }

    found_types = set(
        manifest["dataset_type"]
    )

    missing_types = (
        expected_types - found_types
    )

    if missing_types:
        raise ValueError(
            "Missing dataset types: "
            f"{sorted(missing_types)}"
        )

    for dataset_type in expected_types:
        years_found = set(
            manifest.loc[
                manifest["dataset_type"].eq(
                    dataset_type
                ),
                "vintage_year",
            ].astype(int)
        )

        missing_years = (
            EXPECTED_YEARS - years_found
        )

        if missing_years:
            raise ValueError(
                f"{dataset_type} is missing "
                f"vintage years: "
                f"{sorted(missing_years)}"
            )

    if not (
        manifest["source_columns"]
        == manifest["header_columns"]
    ).all():
        raise ValueError(
            "At least one source file does not "
            "match its official header."
        )


def main() -> None:
    """Run the Raw-to-Bronze ingestion pipeline."""

    zip_files = sorted(
        RAW_DIRECTORY.glob("*.zip")
    )

    if not zip_files:
        raise FileNotFoundError(
            "No ZIP files were found in data/raw."
        )

    ingestion_run_id = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    headers = load_official_headers(
        HEADER_DIRECTORY
    )

    manifest_rows = []

    for zip_path in zip_files:
        print(
            f"\nProcessing {zip_path.name}"
        )

        archive_rows = ingest_zip_archive(
            zip_path=zip_path,
            bronze_directory=(
                BRONZE_DIRECTORY
            ),
            headers=headers,
            ingestion_run_id=(
                ingestion_run_id
            ),
            chunk_size=250_000,
            compression="snappy",
        )

        manifest_rows.extend(
            archive_rows
        )

    manifest = pd.DataFrame(
        manifest_rows
    )

    if manifest.empty:
        raise ValueError(
            "No supported origination or "
            "performance files were ingested."
        )

    validate_manifest(
        manifest
    )

    MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest.to_csv(
        MANIFEST_PATH,
        index=False,
    )

    print(
        "\nBronze ingestion summary"
    )

    summary = (
        manifest.groupby(
            [
                "dataset_type",
                "vintage_year",
            ],
            as_index=False,
        )
        .agg(
            rows_written=(
                "rows_written",
                "sum",
            ),
            parts_written=(
                "parts_written",
                "sum",
            ),
        )
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nPASS: Bronze ingestion completed "
        "for all expected years."
    )

    print(
        f"Manifest saved to: "
        f"{MANIFEST_PATH}"
    )


if __name__ == "__main__":
    main()