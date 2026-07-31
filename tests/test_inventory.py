from pathlib import Path
import zipfile

import pandas as pd
import pytest

from src.data.inventory import (
    INVENTORY_COLUMNS,
    build_raw_inventory,
    calculate_sha256,
    classify_archive,
    classify_member,
    extract_year_and_quarter,
    inventory_zip_archive,
    matches_any_pattern,
    summarize_inventory,
)

@pytest.fixture
def data_config() -> dict:
    """Provide a minimal data configuration for inventory tests."""

    return {
        "source": {
            "vintage_years": [
                2023,
                2024,
                2025,
            ]
        },
        "files": {
            "archive_extension": ".zip",
            "origination_patterns": [
                "*orig*.txt",
            ],
            "performance_patterns": [
                "*perf*.txt",
                "*svcg*.txt",
            ],
            "ignored_patterns": [
                "*crt*",
                "*deal*",
                "*readme*",
                "*.pdf",
            ],
        },
    }


def test_calculate_sha256(
    tmp_path: Path,
) -> None:
    """Confirm that the same file always produces the same checksum."""

    file_path = tmp_path / "sample.txt"

    file_path.write_text(
        "MortgageGuard AI",
        encoding="utf-8",
    )

    first_checksum = calculate_sha256(
        file_path
    )

    second_checksum = calculate_sha256(
        file_path
    )

    assert first_checksum == second_checksum
    assert len(first_checksum) == 64


def test_calculate_sha256_missing_file(
    tmp_path: Path,
) -> None:
    """Confirm that a missing source file raises a clear error."""

    missing_file = (
        tmp_path
        / "missing.txt"
    )

    with pytest.raises(
        FileNotFoundError,
        match="Source file was not found",
    ):
        calculate_sha256(
            missing_file
        )


def test_matches_any_pattern() -> None:
    """Confirm that configured wildcard patterns are applied."""

    result = matches_any_pattern(
        file_name="sample_orig_2023.txt",
        patterns=[
            "*orig*.txt",
        ],
    )

    assert result is True


def test_classify_origination_member() -> None:
    """Confirm that an origination source file is identified."""

    result = classify_member(
        member_name="sample_orig_2023.txt",
        origination_patterns=[
            "*orig*.txt",
        ],
        performance_patterns=[
            "*perf*.txt",
            "*svcg*.txt",
        ],
    )

    assert result == "origination"


def test_classify_performance_member() -> None:
    """Confirm that a performance source file is identified."""

    result = classify_member(
        member_name="sample_perf_2023.txt",
        origination_patterns=[
            "*orig*.txt",
        ],
        performance_patterns=[
            "*perf*.txt",
            "*svcg*.txt",
        ],
    )

    assert result == "performance"


def test_classify_older_performance_name() -> None:
    """Confirm support for an older servicing filename convention."""

    result = classify_member(
        member_name="sample_svcg_2023.txt",
        origination_patterns=[
            "*orig*.txt",
        ],
        performance_patterns=[
            "*perf*.txt",
            "*svcg*.txt",
        ],
    )

    assert result == "performance"


def test_classify_ignored_member() -> None:
    """Confirm that ignored files are excluded before ingestion."""

    result = classify_member(
        member_name="dataset_readme.pdf",
        origination_patterns=[
            "*orig*.txt",
        ],
        performance_patterns=[
            "*perf*.txt",
        ],
        ignored_patterns=[
            "*readme*",
            "*.pdf",
        ],
    )

    assert result == "ignored"


def test_extract_expected_vintage_year() -> None:
    """Confirm that a configured vintage year is extracted."""

    result = extract_vintage_year(
        text="sample_orig_2024.txt",
        expected_years=[
            2023,
            2024,
            2025,
        ],
    )

    assert result == 2024


def test_reject_unexpected_vintage_year() -> None:
    """Confirm that a year outside the project scope is rejected."""

    result = extract_vintage_year(
        text="sample_orig_2022.txt",
        expected_years=[
            2023,
            2024,
            2025,
        ],
    )

    assert result is None


def test_classify_complete_sflld_archive(
    tmp_path: Path,
) -> None:
    """Confirm that an archive with both source types is valid."""

    archive_path = (
        tmp_path
        / "sample_2023.zip"
    )

    result = classify_archive(
        archive_path=archive_path,
        member_types=[
            "origination",
            "performance",
        ],
    )

    assert result == "sflld_candidate"


def test_classify_incomplete_sflld_archive(
    tmp_path: Path,
) -> None:
    """Confirm that an archive missing one source type is incomplete."""

    archive_path = (
        tmp_path
        / "sample_2024.zip"
    )

    result = classify_archive(
        archive_path=archive_path,
        member_types=[
            "origination",
        ],
    )

    assert result == (
        "incomplete_sflld_candidate"
    )


def test_classify_crt_archive(
    tmp_path: Path,
) -> None:
    """Confirm that CRT disclosure data is detected and rejected."""

    archive_path = (
        tmp_path
        / "fre-crt-2026-07.zip"
    )

    result = classify_archive(
        archive_path=archive_path,
        member_types=[
            "other",
        ],
    )

    assert result == "crt_deal_disclosure"


def test_inventory_zip_archive(
    tmp_path: Path,
    data_config: dict,
) -> None:
    """Confirm that a valid Freddie Mac archive is inventoried."""

    archive_path = (
        tmp_path
        / "sample_2023.zip"
    )

    with zipfile.ZipFile(
        archive_path,
        mode="w",
    ) as archive:
        archive.writestr(
            "sample_orig_2023.txt",
            (
                "L001|720|250000\n"
                "L002|680|180000\n"
            ),
        )

        archive.writestr(
            "sample_perf_2023.txt",
            (
                "L001|202301|250000|0\n"
                "L001|202302|249500|0\n"
                "L002|202301|180000|1\n"
            ),
        )

        archive.writestr(
            "dataset_readme.pdf",
            "Reference documentation",
        )

    rows = inventory_zip_archive(
        archive_path=archive_path,
        data_config=data_config,
    )

    inventory = pd.DataFrame(
        rows
    )

    assert len(inventory) == 3

    assert set(
        inventory["member_type"]
    ) == {
        "origination",
        "performance",
        "ignored",
    }

    assert (
        inventory["archive_family"]
        .eq("sflld_candidate")
        .all()
    )

    assert (
        inventory["vintage_year"]
        .eq(2023)
        .all()
    )

    assert inventory[
        "source_sha256"
    ].str.len().eq(64).all()


def test_inventory_invalid_zip_archive(
    tmp_path: Path,
    data_config: dict,
) -> None:
    """Confirm that an invalid ZIP archive raises a clear error."""

    archive_path = (
        tmp_path
        / "sample_2023.zip"
    )

    archive_path.write_text(
        "This is not a valid ZIP archive.",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Invalid ZIP archive",
    ):
        inventory_zip_archive(
            archive_path=archive_path,
            data_config=data_config,
        )


def test_build_raw_inventory(
    tmp_path: Path,
    data_config: dict,
) -> None:
    """Confirm that all supported raw files are inventoried."""

    archive_path = (
        tmp_path
        / "sample_2024.zip"
    )

    with zipfile.ZipFile(
        archive_path,
        mode="w",
    ) as archive:
        archive.writestr(
            "sample_orig_2024.txt",
            "L001|740\n",
        )

        archive.writestr(
            "sample_perf_2024.txt",
            "L001|202401|0\n",
        )

    loose_file = (
        tmp_path
        / "sample_orig_2025.txt"
    )

    loose_file.write_text(
        "L002|710\n",
        encoding="utf-8",
    )

    inventory = build_raw_inventory(
        raw_directory=tmp_path,
        data_config=data_config,
    )

    assert list(
        inventory.columns
    ) == INVENTORY_COLUMNS

    assert len(inventory) == 3

    assert set(
        inventory["vintage_year"]
        .dropna()
        .astype(int)
    ) == {
        2024,
        2025,
    }


def test_build_inventory_missing_directory(
    tmp_path: Path,
    data_config: dict,
) -> None:
    """Confirm that a missing raw directory raises a clear error."""

    missing_directory = (
        tmp_path
        / "missing_raw"
    )

    with pytest.raises(
        FileNotFoundError,
        match="Raw data directory was not found",
    ):
        build_raw_inventory(
            raw_directory=missing_directory,
            data_config=data_config,
        )


def test_summarize_inventory() -> None:
    """Confirm that inventory records are summarized by type and year."""

    inventory = pd.DataFrame(
        [
            {
                "source_archive": "sample_2023.zip",
                "source_sha256": "a" * 64,
                "archive_size_mb": 10.0,
                "archive_family": "sflld_candidate",
                "member_name": "sample_orig_2023.txt",
                "member_type": "origination",
                "vintage_year": 2023,
                "compressed_size_mb": 1.0,
                "uncompressed_size_mb": 2.0,
            },
            {
                "source_archive": "sample_2023.zip",
                "source_sha256": "a" * 64,
                "archive_size_mb": 10.0,
                "archive_family": "sflld_candidate",
                "member_name": "sample_perf_2023.txt",
                "member_type": "performance",
                "vintage_year": 2023,
                "compressed_size_mb": 4.0,
                "uncompressed_size_mb": 8.0,
            },
        ]
    )

    summary = summarize_inventory(
        inventory
    )

    assert len(summary) == 2

    assert set(
        summary["member_type"]
    ) == {
        "origination",
        "performance",
    }

    assert (
        summary["file_count"]
        .eq(1)
        .all()
    )

    assert (
        summary["uncompressed_size_mb"]
        .sum()
        == 10.0
    )