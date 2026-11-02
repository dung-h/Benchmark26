"""Tests for provenance descriptors, source pins, and protocol schemas."""
import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_data_sources_schema():
    path = ROOT / "upstream/data_sources.json"
    assert path.is_file(), "upstream/data_sources.json must exist"
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert "schema_version" in data
    assert "repositories" in data
    assert "pin_validation" in data
    assert "assets" in data

    repos = data["repositories"]
    assert "mali" in repos
    assert "qonductor" in repos
    assert "qpack" in repos

    assert len(data["assets"]) >= 200, "Should pin over 200 original source files"
    for asset in data["assets"][:10]:
        assert "source" in asset
        assert "url" in asset
        assert "sha256" in asset


def test_qualification_provenance_and_hash():
    prov_path = ROOT / "upstream/qualification_provenance.json"
    assert prov_path.is_file(), "qualification_provenance.json must exist"
    with prov_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    target_file = ROOT / meta["file"]
    assert target_file.is_file(), f"Target file {meta['file']} must exist"

    # Verify sha256 matches
    hasher = hashlib.sha256()
    with target_file.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    actual_hash = hasher.hexdigest()

    assert actual_hash == meta["sha256"], "Screening CSV SHA256 must match provenance ledger"


def test_component_protocols_validity():
    protocol_dir = ROOT / "benchmark_v1/protocol"
    assert protocol_dir.is_dir(), "benchmark_v1/protocol must exist"

    protocol_files = list(protocol_dir.glob("*.json"))
    assert len(protocol_files) >= 6, f"Expected at least 6 protocol files, found {len(protocol_files)}"

    for proto_file in protocol_files:
        with proto_file.open("r", encoding="utf-8") as f:
            content = json.load(f)
        assert isinstance(content, dict), f"{proto_file.name} must be a JSON object"
        assert "contract_id" in content or "protocol_id" in content, (
            f"{proto_file.name} must declare contract_id or protocol_id"
        )
