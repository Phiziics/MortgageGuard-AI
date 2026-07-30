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
    "source_path",
    "source_sha256",
    "archive_size_mb",
    "archive_family",
    "member_name",
    "member_type",
    "vintage_year",
    "vintage_quarter",
    "compressed_size_mb",
    "uncompressed_size_mb",
]


def calculate_sha256(
    file_path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Calculate the SHA256 checksum of a source file."""

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
    """Return True when a filename matches a configured pattern."""

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
    """Classify a file contained inside an archive."""

    ignored_patterns = ignored_patterns or []

    if matches_any_pattern(
        member_name,
        ignored_patterns,
    ):
        return "ignored"

    # Check performance first because its filename is more specific.
    if matches_any_pattern(
        member_name,
        performance_patterns,
    ):
        return "performance"

    if matches_any_pattern(
        member_name,
        origination_patterns,
    ):
        return "origination"

    if Path(member_name).suffix.lower() == ".zip":
        return "nested_archive"

    return "other"


def extract_year_and_quarter(
    text: str,
    expected_years: list[int] | None = None,
) -> tuple[int | None, int | None]:
    """Extract a vintage year and quarter from Freddie Mac filenames."""

    match = re.search(
        r"(?<!\d)(20\d{2})Q([1-4])(?!\d)",
        text,
        flags=re.IGNORECASE,
    )

    if match is None:
        return None, None

    year = int(match.group(1))
    quarter = int(match.group(2))

    if expected_years and year not in set(expected_years):
        return None, None

    return year, quarter


def classify_archive(
    archive_path: Path,
    member_names: list[str],
    member_types: list[str],
) -> str:
    """Classify a ZIP archive based on its name and contents."""

    archive_name = archive_path.name.lower()

    if "crt" in archive_name or "deal" in archive_name:
        return "crt_deal_disclosure"

    detected_types = set(member_types)

    if {
        "origination",
        "performance",
    }.issubset(detected_types):
        return "sflld_quarter_archive"

    if (
        "origination" in detected_types
        or "performance" in detected_types
    ):
        return "incomplete_sflld_quarter_archive"

    # The outer yearly package contains quarterly ZIP archives.
    if member_names and all(
        Path(member_name).suffix.lower() == ".zip"
        for member_name in member_names
    ):
        return "outer_year_package"

    return "unknown"


def inventory_zip_archive(
    archive_path: Path,
    raw_directory: Path,
    data_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Inspect one ZIP archive without extracting its TXT files."""

    if not archive_path.exists():
        raise FileNotFoundError(
            f"Archive was not found: {archive_path}"
        )

    file_config = data_config["files"]
    expected_years = data_config["source"]["vintage_years"]

    source_checksum = calculate_sha256(archive_path)

    archive_size_mb = round(
        archive_path.stat().st_size / 1_048_576,
        2,
    )

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

            member_names = [
                member.filename
                for member in members
            ]

            member_types = [
                classify_member(
                    member_name=member.filename,
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
                for member in members
            ]

            archive_family = classify_archive(
                archive_path=archive_path,
                member_names=member_names,
                member_types=member_types,
            )

            rows: list[dict[str, Any]] = []

            for member, member_type in zip(
                members,
                member_types,
            ):
                year, quarter = extract_year_and_quarter(
                    text=(
                        f"{archive_path.name} "
                        f"{member.filename}"
                    ),
                    expected_years=expected_years,
                )

                rows.append(
                    {
                        "source_archive": archive_path.name,
                        "source_path": str(
                            archive_path.relative_to(
                                raw_directory
                            )
                        ),
                        "source_sha256": source_checksum,
                        "archive_size_mb": archive_size_mb,
                        "archive_family": archive_family,
                        "member_name": member.filename,
                        "member_type": member_type,
                        "vintage_year": year,
                        "vintage_quarter": quarter,
                        "compressed_size_mb": round(
                            member.compress_size / 1_048_576,
                            2,
                        ),
                        "uncompressed_size_mb": round(
                            member.file_size / 1_048_576,
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
    raw_directory: Path,
    data_config: dict[str, Any],
) -> dict[str, Any]:
    """Inventory an already extracted Freddie Mac TXT file."""

    file_config = data_config["files"]

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

    year, quarter = extract_year_and_quarter(
        text=text_path.name,
        expected_years=data_config["source"][
            "vintage_years"
        ],
    )

    file_size_mb = round(
        text_path.stat().st_size / 1_048_576,
        2,
    )

    return {
        "source_archive": text_path.name,
        "source_path": str(
            text_path.relative_to(raw_directory)
        ),
        "source_sha256": calculate_sha256(
            text_path
        ),
        "archive_size_mb": file_size_mb,
        "archive_family": "loose_text_file",
        "member_name": text_path.name,
        "member_type": member_type,
        "vintage_year": year,
        "vintage_quarter": quarter,
        "compressed_size_mb": None,
        "uncompressed_size_mb": file_size_mb,
    }


def build_raw_inventory(
    raw_directory: Path,
    data_config: dict[str, Any],
) -> pd.DataFrame:
    """Build an inventory of all supported raw files recursively."""

    if not raw_directory.exists():
        raise FileNotFoundError(
            f"Raw data directory was not found: "
            f"{raw_directory}"
        )

    rows: list[dict[str, Any]] = []

    # Search all raw-data subfolders for quarterly ZIP archives.
    archive_files = sorted(
        raw_directory.rglob("*.zip")
    )

    for archive_path in archive_files:
        rows.extend(
            inventory_zip_archive(
                archive_path=archive_path,
                raw_directory=raw_directory,
                data_config=data_config,
            )
        )

    # Also support TXT files that were manually extracted.
    text_files = sorted(
        raw_directory.rglob("*.txt")
    )

    for text_path in text_files:
        rows.append(
            inventory_loose_text_file(
                text_path=text_path,
                raw_directory=raw_directory,
                data_config=data_config,
            )
        )

    return pd.DataFrame(
        rows,
        columns=INVENTORY_COLUMNS,
    )


def summarize_inventory(
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize usable files by vintage, quarter, and dataset type."""

    summary_columns = [
        "vintage_year",
        "vintage_quarter",
        "member_type",
        "file_count",
        "uncompressed_size_mb",
    ]

    if inventory.empty:
        return pd.DataFrame(
            columns=summary_columns
        )

    usable_inventory = inventory.loc[
        inventory["member_type"].isin(
            [
                "origination",
                "performance",
            ]
        )
    ].copy()

    if usable_inventory.empty:
        return pd.DataFrame(
            columns=summary_columns
        )

    summary = (
        usable_inventory.groupby(
            [
                "vintage_year",
                "vintage_quarter",
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
                "vintage_quarter",
                "member_type",
            ]
        )
        .reset_index(drop=True)
    )

    summary["uncompressed_size_mb"] = (
        summary["uncompressed_size_mb"].round(2)
    )

    return summary