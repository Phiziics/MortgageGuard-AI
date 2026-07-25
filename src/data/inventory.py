from __future__ import annotations

import hashlib
import re
import zipfile
from pathlib import Path

import pandas as pd


INVENTORY_COLUMNS = [
    "source_file",
    "source_sha256",
    "dataset_family",
    "vintage_year",
    "member_name",
    "member_type",
    "compressed_mb",
    "uncompressed_mb",
]


def calculate_sha256(file_path: Path) -> str:
    """Calculate a SHA-256 checksum for data lineage and integrity checks."""

    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            sha256.update(chunk)

    return sha256.hexdigest()


def classify_member(member_name: str) -> str:
    """Classify a file as origination, performance, or another file type."""

    name = Path(member_name).name.lower()

    if "orig" in name:
        return "origination"

    if "svcg" in name or "performance" in name or "monthly" in name:
        return "performance"

    return "other"


def extract_vintage_year(text: str) -> int | None:
    """Extract a vintage year from a source or member filename."""

    matches = re.findall(r"\b(20\d{2})\b", text)

    if not matches:
        return None

    return int(matches[0])


def classify_dataset_family(
    source_file: Path,
    member_types: list[str],
) -> str:
    """Identify whether an archive appears to contain the required dataset."""

    source_name = source_file.name.lower()

    if source_name.startswith("fre-crt-"):
        return "crt_deal_disclosure"

    required_types = {"origination", "performance"}

    if required_types.issubset(set(member_types)):
        return "sf_lld_candidate"

    return "unknown"


def inventory_zip_file(zip_path: Path) -> list[dict]:
    """Inspect a ZIP archive without extracting it."""

    rows = []
    checksum = calculate_sha256(zip_path)

    with zipfile.ZipFile(zip_path) as archive:
        members = [
            member
            for member in archive.infolist()
            if not member.is_dir()
        ]

        member_types = [
            classify_member(member.filename)
            for member in members
        ]

        dataset_family = classify_dataset_family(
            zip_path,
            member_types,
        )

        for member, member_type in zip(members, member_types):
            year_text = f"{zip_path.name} {member.filename}"

            rows.append(
                {
                    "source_file": zip_path.name,
                    "source_sha256": checksum,
                    "dataset_family": dataset_family,
                    "vintage_year": extract_vintage_year(year_text),
                    "member_name": member.filename,
                    "member_type": member_type,
                    "compressed_mb": round(
                        member.compress_size / 1_048_576,
                        2,
                    ),
                    "uncompressed_mb": round(
                        member.file_size / 1_048_576,
                        2,
                    ),
                }
            )

    return rows


def inventory_text_file(text_path: Path) -> dict:
    """Inventory an already extracted TXT file."""

    return {
        "source_file": text_path.name,
        "source_sha256": calculate_sha256(text_path),
        "dataset_family": "loose_text_file",
        "vintage_year": extract_vintage_year(text_path.name),
        "member_name": text_path.name,
        "member_type": classify_member(text_path.name),
        "compressed_mb": None,
        "uncompressed_mb": round(
            text_path.stat().st_size / 1_048_576,
            2,
        ),
    }


def build_raw_inventory(raw_directory: Path) -> pd.DataFrame:
    """Build an inventory of all ZIP and TXT files in the raw layer."""

    rows = []

    for zip_path in sorted(raw_directory.glob("*.zip")):
        rows.extend(inventory_zip_file(zip_path))

    for text_path in sorted(raw_directory.glob("*.txt")):
        rows.append(inventory_text_file(text_path))

    return pd.DataFrame(rows, columns=INVENTORY_COLUMNS)