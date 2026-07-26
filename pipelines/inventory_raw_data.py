from pathlib import Path

from src.data.inventory import build_raw_inventory


RAW_DIRECTORY = Path("data/raw")
OUTPUT_FILE = Path("reports/raw_data_inventory.csv")

EXPECTED_YEARS = {2023, 2024, 2025}


def main() -> None:
    """Inspect and validate the files currently stored in the raw layer."""

    inventory = build_raw_inventory(RAW_DIRECTORY)

    if inventory.empty:
        raise FileNotFoundError(
            "No ZIP or TXT files were found in data/raw."
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(OUTPUT_FILE, index=False)

    print("\nRaw data inventory")
    print(
        inventory[
            [
                "source_file",
                "dataset_family",
                "vintage_year",
                "member_type",
                "uncompressed_mb",
            ]
        ].to_string(index=False)
    )

    if (
        inventory["dataset_family"]
        .eq("crt_deal_disclosure")
        .any()
    ):
        print(
            "\nWARNING: CRT deal-disclosure data was detected. "
            "This is not the required SFLLD dataset."
        )

    origination_years = set(
        inventory.loc[
            inventory["member_type"].eq("origination"),
            "vintage_year",
        ].dropna().astype(int)
    )

    performance_years = set(
        inventory.loc[
            inventory["member_type"].eq("performance"),
            "vintage_year",
        ].dropna().astype(int)
    )

    missing_origination = EXPECTED_YEARS - origination_years
    missing_performance = EXPECTED_YEARS - performance_years

    print("\nVintage validation")
    print(f"Origination years found: {sorted(origination_years)}")
    print(f"Performance years found: {sorted(performance_years)}")

    if missing_origination:
        print(
            "Missing origination years: "
            f"{sorted(missing_origination)}"
        )

    if missing_performance:
        print(
            "Missing performance years: "
            f"{sorted(missing_performance)}"
        )

    if not missing_origination and not missing_performance:
        print(
            "\nPASS: Origination and performance files were found "
            "for every expected vintage."
        )

    print(f"\nInventory saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()