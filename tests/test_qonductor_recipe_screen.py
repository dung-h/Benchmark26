"""Tests for Qonductor recipe screening table integrity."""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_qonductor_recipe_screen_table():
    csv_path = ROOT / "upstream/qonductor_recipe_screen.csv"
    assert csv_path.is_file(), "qonductor_recipe_screen.csv must exist"

    expected_headers = [
        "canonical_observation_id",
        "family",
        "candidate_widths",
        "already_in_qualified_panel",
        "archive_match_status",
        "historical_instance_status",
        "recipe_test_status",
        "promoted_to_benchmark",
    ]

    row_count = 0
    unique_ids = set()

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == expected_headers, f"Header mismatch: {reader.fieldnames}"

        for row in reader:
            row_count += 1
            cid = row["canonical_observation_id"]
            assert cid not in unique_ids, f"Duplicate ID: {cid}"
            unique_ids.add(cid)
            assert row["recipe_test_status"] in (
                "REPEATABLE_UNDER_TEST",
                "VARIES_OR_ERROR",
                "NOT_TESTED_OR_NO_UNIQUE_WIDTH",
                "NOT_CHECKED_WIDTH_GUARD",
            ), f"Invalid status: {row['recipe_test_status']}"

    assert row_count == 4482, f"Expected 4482 rows, found {row_count}"
