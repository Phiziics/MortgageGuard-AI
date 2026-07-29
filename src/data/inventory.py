from __future__ import annotations

import fnmatch
import hashlib
import re
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd


INVENTORY_COLUMNS = [
    "source_archive",
    "source_sha256",
    "archive_size_mb",
    "archive_family",
    "member_name",
    "member_type",
    "vintage_year",
    "compressed_size_mb",
    "uncompressed_size_mb",
]


def calculate_sha256(
    file_path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Calculate a SHA256 checksum for a source file.

    The checksum allows the project to detect whether a raw
    archive changed between ingestion runs.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Source file was not found: {file_path}"
        )

    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


def matches_any_pattern(
    file_name: str,
    patterns: list[str],
) -> bool:
    """Return True when a filename matches any configured pattern."""

    normalized_name = Path(file_name).name.lower()

    return any(
        fnmatch.fnmatch(
            normalized_name,
            pattern.lower(),
        )
        for pattern in patterns
    )


def classify_member(
    member_name: str,
    origination_patterns: list[str],
    performance_patterns: list[str],
    ignored_patterns: list[str] | None = None,
) -> str:
    """Classify a file inside an archive.

    Possible values:

    origination
    performance
    ignored
    other
    """

    ignored_patterns = ignored_patterns or []

    if matches_any_pattern(
        member_name,
        ignored_patterns,
    ):
        return "ignored"

    if matches_any_pattern(
        member_name,
        origination_patterns,
    ):
        return "origination"

    if matches_any_pattern(
        member_name,
        performance_patterns,
    ):
        return "performance"

    return "other"


def extract_vintage_year(
    text: str,
    expected_years: list[int] | None = None,
) -> int | None:
    """Extract a Freddie Mac vintage year from a filename.

    When expected years are supplied, only configured project
    years are accepted.
    """

    detected_years = [
        int(year)
        for year in re.findall(
            r"(?<!\d)(20\d{2})(?!\d)",
            text,
        )
    ]

    if expected_years:
        expected_year_set = set(
            expected_years
        )

        matching_years = [
            year
            for year in detected_years
            if year in expected_year_set
        ]

        if matching_years:
            return matching_years[0]

        return None

    if detected_years:
        return detected_years[0]

    return None


def classify_archive(
    archive_path: Path,
    member_types: list[str],
) -> str:
    """Classify the overall archive based on its name and contents."""

    archive_name = archive_path.name.lower()

    # Prevent accidental ingestion of CRT deal disclosure data.
    if (
        "crt" in archive_name
        or "deal" in archive_name
    ):
        return "crt_deal_disclosure"

    detected_types = set(
        member_types
    )

    required_types = {
        "origination",
        "performance",
    }

    if required_types.issubset(
        detected_types
    ):
        return "sflld_candidate"

    if (
        "origination" in detected_types
        or "performance" in detected_types
    ):
        return "incomplete_sflld_candidate"

    return "unknown"


def inventory_zip_archive(
    archive_path: Path,
    data_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Inspect one ZIP archive without extracting it."""

    if not archive_path.exists():
        raise FileNotFoundError(
            f"Archive was not found: {archive_path}"
        )

    file_config = data_config["files"]
    expected_years = data_config["source"][
        "vintage_years"
    ]

    origination_patterns = file_config[
        "origination_patterns"
    ]

    performance_patterns = file_config[
        "performance_patterns"
    ]

    ignored_patterns = file_config.get(
        "ignored_patterns",
        [],
    )

    source_checksum = calculate_sha256(
        archive_path
    )

    archive_size_mb = round(
        archive_path.stat().st_size
        / 1_048_576,
        2,
    )

    rows: list[dict[str, Any]] = []

    try:
        with zipfile.ZipFile(
            archive_path,
            mode="r",
        ) as archive:
            members = [
                member
                for member in archive.infolist()
                if not member.is_dir()
            ]

            member_types = [
                classify_member(
                    member_name=member.filename,
                    origination_patterns=(
                        origination_patterns
                    ),
                    performance_patterns=(
                        performance_patterns
                    ),
                    ignored_patterns=(
                        ignored_patterns
                    ),
                )
                for member in members
            ]

            archive_family = classify_archive(
                archive_path=archive_path,
                member_types=member_types,
            )

            for member, member_type in zip(
                members,
                member_types,
            ):
                combined_name = (
                    f"{archive_path.name} "
                    f"{member.filename}"
                )

                vintage_year = (
                    extract_vintage_year(
                        text=combined_name,
                        expected_years=(
                            expected_years
                        ),
                    )
                )

                rows.append(
                    {
                        "source_archive": (
                            archive_path.name
                        ),
                        "source_sha256": (
                            source_checksum
                        ),
                        "archive_size_mb": (
                            archive_size_mb
                        ),
                        "archive_family": (
                            archive_family
                        ),
                        "member_name": (
                            member.filename
                        ),
                        "member_type": (
                            member_type
                        ),
                        "vintage_year": (
                            vintage_year
                        ),
                        "compressed_size_mb": round(
                            member.compress_size
                            / 1_048_576,
                            2,
                        ),
                        "uncompressed_size_mb": round(
                            member.file_size
                            / 1_048_576,
                            2,
                        ),
                    }
                )

    except zipfile.BadZipFile as error:
        raise ValueError(
            f"Invalid ZIP archive: {archive_path}"
        ) from error

    return rows


def inventory_loose_text_file(
    text_path: Path,
    data_config: dict[str, Any],
) -> dict[str, Any]:
    """Inventory a text file stored directly in the raw folder."""

    file_config = data_config["files"]
    expected_years = data_config["source"][
        "vintage_years"
    ]

    member_type = classify_member(
        member_name=text_path.name,
        origination_patterns=file_config[
            "origination_patterns"
        ],
        performance_patterns=file_config[
            "performance_patterns"
        ],
        ignored_patterns=file_config.get(
            "ignored_patterns",
            [],
        ),
    )

    return {
        "source_archive": text_path.name,
        "source_sha256": calculate_sha256(
            text_path
        ),
        "archive_size_mb": round(
            text_path.stat().st_size
            / 1_048_576,
            2,
        ),
        "archive_family": "loose_text_file",
        "member_name": text_path.name,
        "member_type": member_type,
        "vintage_year": extract_vintage_year(
            text=text_path.name,
            expected_years=expected_years,
        ),
        "compressed_size_mb": None,
        "uncompressed_size_mb": round(
            text_path.stat().st_size
            / 1_048_576,
            2,
        ),
    }


def build_raw_inventory(
    raw_directory: Path,
    data_config: dict[str, Any],
) -> pd.DataFrame:
    """Build an inventory for every supported raw file."""

    if not raw_directory.exists():
        raise FileNotFoundError(
            f"Raw data directory was not found: "
            f"{raw_directory}"
        )

    archive_extension = data_config[
        "files"
    ].get(
        "archive_extension",
        ".zip",
    )

    rows: list[dict[str, Any]] = []

    # Inspect ZIP archives without extracting them.
    archive_files = sorted(
        raw_directory.glob(
            f"*{archive_extension}"
        )
    )

    for archive_path in archive_files:
        archive_rows = (
            inventory_zip_archive(
                archive_path=archive_path,
                data_config=data_config,
            )
        )

        rows.extend(
            archive_rows
        )

    # Also support already extracted source files.
    text_files = sorted(
        raw_directory.glob("*.txt")
    )

    for text_path in text_files:
        rows.append(
            inventory_loose_text_file(
                text_path=text_path,
                data_config=data_config,
            )
        )

    inventory = pd.DataFrame(
        rows,
        columns=INVENTORY_COLUMNS,
    )

    return inventory


def summarize_inventory(
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize detected records by vintage and dataset type."""

    if inventory.empty:
        return pd.DataFrame(
            columns=[
                "vintage_year",
                "member_type",
                "file_count",
                "uncompressed_size_mb",
            ]
        )

    usable_inventory = inventory.loc[
        inventory["member_type"].isin(
            [
                "origination",
                "performance",
            ]
        )
    ].copy()

    summary = (
        usable_inventory.groupby(
            [
                "vintage_year",
                "member_type",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            file_count=(
                "member_name",
                "count",
            ),
            uncompressed_size_mb=(
                "uncompressed_size_mb",
                "sum",
            ),
        )
        .sort_values(
            [
                "vintage_year",
                "member_type",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    summary[
        "uncompressed_size_mb"
    ] = summary[
        "uncompressed_size_mb"
    ].round(2)

    return summary