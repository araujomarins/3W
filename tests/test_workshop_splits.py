from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPT = (
    Path(__file__).parents[1]
    / "toolkit"
    / "demos"
    / "workshop"
    / "prepare_workshop_splits.py"
)
_WORKSHOP = _SCRIPT.parent
_SPEC = importlib.util.spec_from_file_location("prepare_workshop_splits", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _fake_dataset(root: Path) -> None:
    for event_class in range(10):
        class_dir = root / str(event_class)
        class_dir.mkdir(parents=True)
        for instance_number in range(10):
            (class_dir / f"SIMULATED_{instance_number:05d}.parquet").touch()

    # Both paths represent the same real well-day leakage group.
    (root / "0" / "WELL-00001_20250101000000.parquet").touch()
    (root / "1" / "WELL-00001_20250101120000.parquet").touch()


def test_matrix_is_complete_deterministic_and_group_safe(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _fake_dataset(dataset_root)
    records = _MODULE.discover_instances(dataset_root)

    first = _MODULE.assign_validation_splits(records)
    second = _MODULE.assign_validation_splits(records)

    assert first == second
    assert len(first) == len(records)
    assert set(first.values()) == {1, 2, 3, 4, 5}
    assert (
        first["0/WELL-00001_20250101000000.parquet"]
        == first["1/WELL-00001_20250101120000.parquet"]
    )


def test_matrix_converts_to_five_disjoint_validation_lists(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _fake_dataset(dataset_root)
    records = _MODULE.discover_instances(dataset_root)
    assignment = _MODULE.assign_validation_splits(records)
    matrix = tmp_path / "instance_split_matrix.csv"
    _MODULE.write_matrix(records, assignment, matrix)

    outputs = _MODULE.matrix_to_validation_files(matrix, tmp_path / "splits")

    assert len(outputs) == 5
    validation_sets = [
        set(path.read_text(encoding="utf-8").splitlines()) for path in outputs
    ]
    assert sum(map(len, validation_sets)) == len(records)
    assert set.union(*validation_sets) == {record.path for record in records}
    assert all(
        left.isdisjoint(right)
        for index, left in enumerate(validation_sets)
        for right in validation_sets[index + 1 :]
    )


def test_checked_in_v2_matrix_matches_validation_lists() -> None:
    rows = _MODULE.read_matrix(_WORKSHOP / "instance_split_matrix.csv")
    expected_by_split = {
        split_number: {
            row["instance"] for row in rows if row[f"split_{split_number}"] == "1"
        }
        for split_number in range(1, 6)
    }
    actual_by_split = {
        split_number: set(
            (_WORKSHOP / "splits" / f"validation_split_{split_number}.txt")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        for split_number in range(1, 6)
    }

    assert len(rows) == 2228
    assert actual_by_split == expected_by_split
    assert [len(actual_by_split[split]) for split in range(1, 6)] == [
        446,
        447,
        444,
        446,
        445,
    ]
