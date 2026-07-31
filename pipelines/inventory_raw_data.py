from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import config, resolve_project_path
from src.data.inventory import (
    build_raw_inventory,
    summarize_inventory,
)


EXPECTED_QUARTERS = {
    1,
    2,
    3,
    4,
}


def validate_archive_families(
    inventory: pd.DataFrame,
) -> None:
    """Reject incorrect, corrupt, incomplete, or unknown archives."""

    # Reject CRT deal disclosure files.
    crt_records = inventory.loc[
        inventory["archive_family"].eq(
            "crt_deal_disclosure"
        )
    ]

    if not crt_records.empty:
        detected_files = sorted(
            crt_records["source_path"]
            .dropna()
            .unique()
            .tolist()
        )

        raise ValueError(
            "CRT deal disclosure data was detected. "
            "MortgageGuard AI requires Freddie Mac "
            "Single-Family Loan-Level historical data. "
            f"Remove these files: {detected_files}"
        )

    # Reject quarterly archives missing either source file.
    incomplete_records = inventory.loc[
        inventory["archive_family"].eq(
            "incomplete_sflld_quarter_archive"
        )
    ]

    if not incomplete_records.empty:
        detected_files = sorted(
            incomplete_records["source_path"]
            .dropna()
            .unique()
            .tolist()
        )

        raise ValueError(
            "Incomplete quarterly archives were detected. "
            "Each quarterly archive must contain both an "
            "origination file and a performance file. "
            f"Incomplete archives: {detected_files}"
        )

    # Unknown ZIP files should be reviewed before ingestion.
    unknown_records = inventory.loc[
        inventory["archive_family"].eq(
            "unknown"
        )
    ]

    if not unknown_records.empty:
        detected_files = sorted(
            unknown_records["source_path"]
            .dropna()
            .unique()
            .tolist()
        )

        raise ValueError(
            "Unrecognized ZIP archives were detected. "
            f"Review these files: {detected_files}"
        )


def get_usable_inventory(
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    """Return only quarterly origination and performance records."""

    usable_inventory = inventory.loc[
        inventory["archive_family"].eq(
            "sflld_quarter_archive"
        )
        & inventory["member_type"].isin(
            [
                "origination",
                "performance",
            ]
        )
    ].copy()

    if usable_inventory.empty:
        raise ValueError(
            "No usable Freddie Mac quarterly archives were found. "
            "Extract the outer yearly ZIP packages so the quarterly "
            "ZIP files are available under data/raw."
        )

    return usable_inventory


def validate_year_and_quarter_values(
    usable_inventory: pd.DataFrame,
) -> None:
    """Confirm every usable file has a recognized year and quarter."""

    missing_period = usable_inventory.loc[
        usable_inventory[
            [
                "vintage_year",
                "vintage_quarter",
            ]
        ].isna().any(axis=1)
    ]

    if not missing_period.empty:
        missing_files = sorted(
            missing_period["member_name"]
            .dropna()
            .unique()
            .tolist()
        )

        raise ValueError(
            "A configured year or quarter could not be identified "
            f"for these files: {missing_files}"
        )


def validate_expected_years(
    usable_inventory: pd.DataFrame,
    expected_years: list[int],
) -> None:
    """Confirm only the configured vintage years are included."""

    expected_year_set = set(
        expected_years
    )

    detected_years = set(
        usable_inventory["vintage_year"]
        .dropna()
        .astype(int)
        .tolist()
    )

    missing_years = (
        expected_year_set
        - detected_years
    )

    unexpected_years = (
        detected_years
        - expected_year_set
    )

    if missing_years:
        raise ValueError(
            "Missing Freddie Mac vintage years: "
            f"{sorted(missing_years)}"
        )

    if unexpected_years:
        raise ValueError(
            "Unexpected Freddie Mac vintage years were detected: "
            f"{sorted(unexpected_years)}"
        )


def validate_quarter_coverage(
    usable_inventory: pd.DataFrame,
    expected_years: list[int],
) -> None:
    """Confirm that all four quarters exist for every configured year."""

    problems: list[str] = []

    for year in expected_years:
        year_records = usable_inventory.loc[
            usable_inventory[
                "vintage_year"
            ].eq(year)
        ]

        for member_type in [
            "origination",
            "performance",
        ]:
            detected_quarters = set(
                year_records.loc[
                    year_records[
                        "member_type"
                    ].eq(member_type),
                    "vintage_quarter",
                ]
                .dropna()
                .astype(int)
                .tolist()
            )

            missing_quarters = (
                EXPECTED_QUARTERS
                - detected_quarters
            )

            unexpected_quarters = (
                detected_quarters
                - EXPECTED_QUARTERS
            )

            if missing_quarters:
                problems.append(
                    f"{year} {member_type} missing quarters "
                    f"{sorted(missing_quarters)}"
                )

            if unexpected_quarters:
                problems.append(
                    f"{year} {member_type} has unexpected quarters "
                    f"{sorted(unexpected_quarters)}"
                )

    if problems:
        raise ValueError(
            "Quarter coverage validation failed: "
            + "; ".join(problems)
        )


def validate_one_file_per_quarter(
    usable_inventory: pd.DataFrame,
) -> None:
    """Confirm each year, quarter, and source type appears exactly once."""

    file_counts = (
        usable_inventory.groupby(
            [
                "vintage_year",
                "vintage_quarter",
                "member_type",
            ],
            as_index=False,
        )
        .agg(
            file_count=(
                "member_name",
                "count",
            )
        )
    )

    duplicate_sources = file_counts.loc[
        file_counts["file_count"].ne(1)
    ]

    if not duplicate_sources.empty:
        raise ValueError(
            "Each vintage year, quarter, and source type must "
            "appear exactly once. Invalid groups: "
            f"{duplicate_sources.to_dict(orient='records')}"
        )


def validate_expected_file_count(
    usable_inventory: pd.DataFrame,
    expected_years: list[int],
) -> None:
    """Confirm the total number of usable source files."""

    source_types_per_quarter = 2
    quarters_per_year = 4

    expected_file_count = (
        len(expected_years)
        * quarters_per_year
        * source_types_per_quarter
    )

    actual_file_count = len(
        usable_inventory
    )

    if actual_file_count != expected_file_count:
        raise ValueError(
            "Unexpected number of usable Freddie Mac source files. "
            f"Expected {expected_file_count}, "
            f"found {actual_file_count}."
        )


def validate_inventory(
    inventory: pd.DataFrame,
    expected_years: list[int],
) -> pd.DataFrame:
    """Run all raw-data inventory validation controls."""

    if inventory.empty:
        raise FileNotFoundError(
            "No ZIP or TXT files were found under data/raw."
        )

    validate_archive_families(
        inventory
    )

    usable_inventory = get_usable_inventory(
        inventory
    )

    validate_year_and_quarter_values(
        usable_inventory
    )

    validate_expected_years(
        usable_inventory=usable_inventory,
        expected_years=expected_years,
    )

    validate_quarter_coverage(
        usable_inventory=usable_inventory,
        expected_years=expected_years,
    )

    validate_one_file_per_quarter(
        usable_inventory
    )

    validate_expected_file_count(
        usable_inventory=usable_inventory,
        expected_years=expected_years,
    )

    return usable_inventory


def save_inventory_reports(
    inventory: pd.DataFrame,
    usable_inventory: pd.DataFrame,
    summary: pd.DataFrame,
) -> tuple[Path, Path, Path]:
    """Save detailed, usable, and summarized inventory reports."""

    detailed_output = resolve_project_path(
        config.data["outputs"][
            "raw_inventory_report"
        ]
    )

    usable_output = (
        detailed_output.parent
        / "raw_data_inventory_usable.csv"
    )

    summary_output = (
        detailed_output.parent
        / "raw_data_inventory_summary.csv"
    )

    detailed_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    inventory.to_csv(
        detailed_output,
        index=False,
    )

    usable_inventory.to_csv(
        usable_output,
        index=False,
    )

    summary.to_csv(
        summary_output,
        index=False,
    )

    return (
        detailed_output,
        usable_output,
        summary_output,
    )


def print_outer_packages(
    inventory: pd.DataFrame,
) -> None:
    """Display outer yearly packages separately from usable archives."""

    outer_packages = inventory.loc[
        inventory["archive_family"].eq(
            "outer_year_package"
        ),
        [
            "source_archive",
            "source_path",
            "member_name",
        ],
    ]

    if outer_packages.empty:
        return

    print(
        "\nOuter yearly packages detected"
    )

    print(
        outer_packages.to_string(
            index=False
        )
    )

    print(
        "\nThe outer packages are preserved for lineage. "
        "Their nested members are not used directly for ingestion."
    )


def print_usable_inventory(
    usable_inventory: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    """Print the validated quarterly source inventory."""

    print(
        "\nValidated quarterly source files"
    )

    print(
        usable_inventory[
            [
                "source_path",
                "member_name",
                "member_type",
                "vintage_year",
                "vintage_quarter",
                "uncompressed_size_mb",
            ]
        ].sort_values(
            [
                "vintage_year",
                "vintage_quarter",
                "member_type",
            ]
        ).to_string(
            index=False
        )
    )

    print(
        "\nInventory summary"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    total_uncompressed_mb = (
        usable_inventory[
            "uncompressed_size_mb"
        ]
        .fillna(0)
        .sum()
    )

    print(
        "\nTotal usable uncompressed source size: "
        f"{total_uncompressed_mb:,.2f} MB"
    )

    print(
        "Total usable source files: "
        f"{len(usable_inventory)}"
    )


def main() -> None:
    """Run the Freddie Mac raw-data inventory pipeline."""

    expected_years = (
        config.vintage_years
    )

    print(
        f"Project: {config.project_name}"
    )

    print(
        "Expected vintage years: "
        f"{expected_years}"
    )

    print(
        "Expected quarters per year: "
        f"{sorted(EXPECTED_QUARTERS)}"
    )

    print(
        "Raw data directory: "
        f"{config.paths.raw_data}"
    )

    inventory = build_raw_inventory(
        raw_directory=(
            config.paths.raw_data
        ),
        data_config=config.data,
    )

    usable_inventory = validate_inventory(
        inventory=inventory,
        expected_years=expected_years,
    )

    summary = summarize_inventory(
        usable_inventory
    )

    (
        detailed_output,
        usable_output,
        summary_output,
    ) = save_inventory_reports(
        inventory=inventory,
        usable_inventory=usable_inventory,
        summary=summary,
    )

    print_outer_packages(
        inventory
    )

    print_usable_inventory(
        usable_inventory=usable_inventory,
        summary=summary,
    )

    print(
        "\nPASS: Raw Freddie Mac inventory validation "
        "completed successfully."
    )

    print(
        "Detailed inventory: "
        f"{detailed_output}"
    )

    print(
        "Usable inventory: "
        f"{usable_output}"
    )

    print(
        "Inventory summary: "
        f"{summary_output}"
    )


if __name__ == "__main__":
    main()