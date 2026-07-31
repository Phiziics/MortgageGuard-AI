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
    """Provide a minimal configuration for inventory tests."""

    return {
        "source": {
            "vintage_years": [
                2023,
                2024,
                2025,
            ],
        },
        "files": {
            "origination_patterns": [
                "historical_data_20??Q?.txt",
            ],
            "performance_patterns": [
                "historical_data_time_20??Q?.txt",
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
    """Confirm that the same file produces the same checksum."""

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


def test_matches_origination_pattern() -> None:
    """Confirm that the real origination filename matches."""

    result = matches_any_pattern(
        file_name=(
            "historical_data_2023Q1.txt"
        ),
        patterns=[
            "historical_data_20??Q?.txt",
        ],
    )

    assert result is True


def test_matches_performance_pattern() -> None:
    """Confirm that the real performance filename matches."""

    result = matches_any_pattern(
        file_name=(
            "historical_data_time_2023Q1.txt"
        ),
        patterns=[
            "historical_data_time_20??Q?.txt",
        ],
    )

    assert result is True


def test_classify_origination_member(
    data_config: dict,
) -> None:
    """Confirm that an origination file is identified."""

    result = classify_member(
        member_name=(
            "historical_data_2023Q1.txt"
        ),
        origination_patterns=data_config[
            "files"
        ]["origination_patterns"],
        performance_patterns=data_config[
            "files"
        ]["performance_patterns"],
        ignored_patterns=data_config[
            "files"
        ]["ignored_patterns"],
    )

    assert result == "origination"


def test_classify_performance_member(
    data_config: dict,
) -> None:
    """Confirm that a monthly performance file is identified."""

    result = classify_member(
        member_name=(
            "historical_data_time_2023Q1.txt"
        ),
        origination_patterns=data_config[
            "files"
        ]["origination_patterns"],
        performance_patterns=data_config[
            "files"
        ]["performance_patterns"],
        ignored_patterns=data_config[
            "files"
        ]["ignored_patterns"],
    )

    assert result == "performance"


def test_classify_nested_archive(
    data_config: dict,
) -> None:
    """Confirm that a quarterly ZIP inside a yearly ZIP is detected."""

    result = classify_member(
        member_name=(
            "historical_data_2023Q1.zip"
        ),
        origination_patterns=data_config[
            "files"
        ]["origination_patterns"],
        performance_patterns=data_config[
            "files"
        ]["performance_patterns"],
        ignored_patterns=data_config[
            "files"
        ]["ignored_patterns"],
    )

    assert result == "nested_archive"


def test_classify_ignored_member(
    data_config: dict,
) -> None:
    """Confirm that documentation files are ignored."""

    result = classify_member(
        member_name="dataset_readme.pdf",
        origination_patterns=data_config[
            "files"
        ]["origination_patterns"],
        performance_patterns=data_config[
            "files"
        ]["performance_patterns"],
        ignored_patterns=data_config[
            "files"
        ]["ignored_patterns"],
    )

    assert result == "ignored"


def test_extract_year_and_quarter() -> None:
    """Confirm that the vintage year and quarter are extracted."""

    result = extract_year_and_quarter(
        text=(
            "historical_data_2024Q3.zip"
        ),
        expected_years=[
            2023,
            2024,
            2025,
        ],
    )

    assert result == (
        2024,
        3,
    )


def test_reject_unexpected_year() -> None:
    """Confirm that years outside the project scope are rejected."""

    result = extract_year_and_quarter(
        text=(
            "historical_data_2022Q4.zip"
        ),
        expected_years=[
            2023,
            2024,
            2025,
        ],
    )

    assert result == (
        None,
        None,
    )


def test_reject_invalid_quarter() -> None:
    """Confirm that only quarters one through four are accepted."""

    result = extract_year_and_quarter(
        text=(
            "historical_data_2023Q5.zip"
        ),
        expected_years=[
            2023,
            2024,
            2025,
        ],
    )

    assert result == (
        None,
        None,
    )


def test_classify_complete_quarter_archive(
    tmp_path: Path,
) -> None:
    """Confirm that a complete quarterly archive is identified."""

    archive_path = (
        tmp_path
        / "historical_data_2023Q1.zip"
    )

    result = classify_archive(
        archive_path=archive_path,
        member_names=[
            "historical_data_2023Q1.txt",
            (
                "historical_data_time_"
                "2023Q1.txt"
            ),
        ],
        member_types=[
            "origination",
            "performance",
        ],
    )

    assert result == (
        "sflld_quarter_archive"
    )


def test_classify_incomplete_quarter_archive(
    tmp_path: Path,
) -> None:
    """Confirm that an incomplete quarterly archive is identified."""

    archive_path = (
        tmp_path
        / "historical_data_2023Q2.zip"
    )

    result = classify_archive(
        archive_path=archive_path,
        member_names=[
            "historical_data_2023Q2.txt",
        ],
        member_types=[
            "origination",
        ],
    )

    assert result == (
        "incomplete_sflld_quarter_archive"
    )


def test_classify_outer_year_package(
    tmp_path: Path,
) -> None:
    """Confirm that a yearly package containing quarterly ZIPs is detected."""

    archive_path = (
        tmp_path
        / "historical_data_2023.zip"
    )

    result = classify_archive(
        archive_path=archive_path,
        member_names=[
            "historical_data_2023Q1.zip",
            "historical_data_2023Q2.zip",
            "historical_data_2023Q3.zip",
            "historical_data_2023Q4.zip",
        ],
        member_types=[
            "nested_archive",
            "nested_archive",
            "nested_archive",
            "nested_archive",
        ],
    )

    assert result == "outer_year_package"


def test_classify_crt_archive(
    tmp_path: Path,
) -> None:
    """Confirm that CRT disclosure data is detected."""

    archive_path = (
        tmp_path
        / "fre-crt-2026-07.zip"
    )

    result = classify_archive(
        archive_path=archive_path,
        member_names=[
            "deal_data.txt",
        ],
        member_types=[
            "other",
        ],
    )

    assert result == "crt_deal_disclosure"


def test_inventory_quarter_archive(
    tmp_path: Path,
    data_config: dict,
) -> None:
    """Confirm that a quarterly archive is inventoried correctly."""

    archive_path = (
        tmp_path
        / "historical_data_2023Q1.zip"
    )

    with zipfile.ZipFile(
        archive_path,
        mode="w",
    ) as archive:
        archive.writestr(
            "historical_data_2023Q1.txt",
            (
                "L001|720|250000\n"
                "L002|680|180000\n"
            ),
        )

        archive.writestr(
            (
                "historical_data_time_"
                "2023Q1.txt"
            ),
            (
                "L001|202301|250000|0\n"
                "L001|202302|249500|0\n"
                "L002|202301|180000|1\n"
            ),
        )

    rows = inventory_zip_archive(
        archive_path=archive_path,
        raw_directory=tmp_path,
        data_config=data_config,
    )

    inventory = pd.DataFrame(
        rows
    )

    assert len(inventory) == 2

    assert set(
        inventory["member_type"]
    ) == {
        "origination",
        "performance",
    }

    assert (
        inventory["archive_family"]
        .eq("sflld_quarter_archive")
        .all()
    )

    assert (
        inventory["vintage_year"]
        .eq(2023)
        .all()
    )

    assert (
        inventory["vintage_quarter"]
        .eq(1)
        .all()
    )

    assert (
        inventory["source_sha256"]
        .str.len()
        .eq(64)
        .all()
    )


def test_inventory_outer_year_package(
    tmp_path: Path,
    data_config: dict,
) -> None:
    """Confirm that an outer yearly archive is identified safely."""

    archive_path = (
        tmp_path
        / "historical_data_2023.zip"
    )

    with zipfile.ZipFile(
        archive_path,
        mode="w",
    ) as archive:
        archive.writestr(
            "historical_data_2023Q1.zip",
            b"nested archive placeholder",
        )

        archive.writestr(
            "historical_data_2023Q2.zip",
            b"nested archive placeholder",
        )

    rows = inventory_zip_archive(
        archive_path=archive_path,
        raw_directory=tmp_path,
        data_config=data_config,
    )

    inventory = pd.DataFrame(
        rows
    )

    assert len(inventory) == 2

    assert (
        inventory["archive_family"]
        .eq("outer_year_package")
        .all()
    )

    assert (
        inventory["member_type"]
        .eq("nested_archive")
        .all()
    )


def test_inventory_invalid_zip(
    tmp_path: Path,
    data_config: dict,
) -> None:
    """Confirm that a corrupt ZIP archive raises an error."""

    archive_path = (
        tmp_path
        / "historical_data_2023Q1.zip"
    )

    archive_path.write_text(
        "This is not a ZIP archive.",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Invalid ZIP archive",
    ):
        inventory_zip_archive(
            archive_path=archive_path,
            raw_directory=tmp_path,
            data_config=data_config,
        )


def test_build_raw_inventory_recursively(
    tmp_path: Path,
    data_config: dict,
) -> None:
    """Confirm that quarterly ZIPs are found inside subfolders."""

    source_folder = (
        tmp_path
        / "historical_data_2024"
    )

    source_folder.mkdir()

    archive_path = (
        source_folder
        / "historical_data_2024Q2.zip"
    )

    with zipfile.ZipFile(
        archive_path,
        mode="w",
    ) as archive:
        archive.writestr(
            "historical_data_2024Q2.txt",
            "L001|740\n",
        )

        archive.writestr(
            (
                "historical_data_time_"
                "2024Q2.txt"
            ),
            "L001|202404|0\n",
        )

    inventory = build_raw_inventory(
        raw_directory=tmp_path,
        data_config=data_config,
    )

    assert list(
        inventory.columns
    ) == INVENTORY_COLUMNS

    assert len(inventory) == 2

    assert (
        inventory["vintage_year"]
        .eq(2024)
        .all()
    )

    assert (
        inventory["vintage_quarter"]
        .eq(2)
        .all()
    )


def test_build_inventory_missing_directory(
    tmp_path: Path,
    data_config: dict,
) -> None:
    """Confirm that a missing raw directory raises an error."""

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
    """Confirm that inventory is summarized by year, quarter, and type."""

    inventory = pd.DataFrame(
        [
            {
                "source_archive": (
                    "historical_data_2023Q1.zip"
                ),
                "source_path": (
                    "historical_data_2023Q1.zip"
                ),
                "source_sha256": "a" * 64,
                "archive_size_mb": 10.0,
                "archive_family": (
                    "sflld_quarter_archive"
                ),
                "member_name": (
                    "historical_data_2023Q1.txt"
                ),
                "member_type": "origination",
                "vintage_year": 2023,
                "vintage_quarter": 1,
                "compressed_size_mb": 1.0,
                "uncompressed_size_mb": 2.0,
            },
            {
                "source_archive": (
                    "historical_data_2023Q1.zip"
                ),
                "source_path": (
                    "historical_data_2023Q1.zip"
                ),
                "source_sha256": "a" * 64,
                "archive_size_mb": 10.0,
                "archive_family": (
                    "sflld_quarter_archive"
                ),
                "member_name": (
                    "historical_data_time_"
                    "2023Q1.txt"
                ),
                "member_type": "performance",
                "vintage_year": 2023,
                "vintage_quarter": 1,
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
        summary["vintage_year"]
        .eq(2023)
        .all()
    )

    assert (
        summary["vintage_quarter"]
        .eq(1)
        .all()
    )

    assert (
        summary["file_count"]
        .eq(1)
        .all()
    )

    assert (
        summary[
            "uncompressed_size_mb"
        ].sum()
        == 10.0
    )