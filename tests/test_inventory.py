from src.data.inventory import (
    classify_dataset_family,
    classify_member,
    extract_vintage_year,
)


def test_classify_origination_file():
    assert (
        classify_member("sample_orig_2024.txt")
        == "origination"
    )


def test_classify_performance_file():
    assert (
        classify_member("sample_svcg_2024.txt")
        == "performance"
    )


def test_extract_vintage_year():
    assert extract_vintage_year("sample_orig_2023.txt") == 2023


def test_detect_crt_dataset(tmp_path):
    archive_path = tmp_path / "fre-crt-2026-07.zip"

    result = classify_dataset_family(
        archive_path,
        ["other"],
    )

    assert result == "crt_deal_disclosure"


def test_detect_sflld_candidate(tmp_path):
    archive_path = tmp_path / "sample_2024.zip"

    result = classify_dataset_family(
        archive_path,
        ["origination", "performance"],
    )

    assert result == "sf_lld_candidate"