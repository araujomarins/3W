"""Organizer-only generator for the fixed 3W workshop split matrix.

This script discovers the complete dataset and performs the grouped, stratified
assignment. Challenge participants receive the resulting CSV, not this
generation step; their starting split assignment must remain fixed.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold

N_SPLITS = 5
DEFAULT_SEED = 42
SPLIT_COLUMNS = tuple(f"split_{number}" for number in range(1, N_SPLITS + 1))
REAL_INSTANCE = re.compile(r"^(WELL-\d+)_(\d{8})")
WORKSHOP_DIR = Path(__file__).resolve().parent
REPO_ROOT = WORKSHOP_DIR.parents[2]
ENV_FILE = REPO_ROOT / ".env"


def load_env_file(path: Path = ENV_FILE) -> None:
    """Load simple KEY=VALUE entries without overriding the shell environment."""

    if not path.is_file():
        return
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not key.isidentifier():
            raise ValueError(f"Invalid .env entry at {path}:{line_number}")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def configured_path(variable: str, default: Path) -> Path:
    """Resolve an environment path relative to the repository root."""

    value = os.getenv(variable)
    path = Path(value).expanduser() if value else default
    return path if path.is_absolute() else REPO_ROOT / path


def configured_int(variable: str, default: int) -> int:
    """Read an integer environment setting with a clear error message."""

    value = os.getenv(variable)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{variable} must be an integer, got {value!r}") from error


@dataclass(frozen=True)
class InstanceRecord:
    """Metadata needed to assign one dataset instance to a split."""

    path: str
    event_class: int
    source: str
    group: str

    @property
    def stratum(self) -> str:
        """Return the class/source stratum used for balancing."""

        return f"{self.event_class}:{self.source}"


def _record(relative_path: Path) -> InstanceRecord:
    """Parse class, source, and leakage group from a relative path."""

    if len(relative_path.parts) != 2 or not relative_path.parent.name.isdigit():
        raise ValueError(
            "Expected paths in '<class>/<instance>.parquet' form, got "
            f"{relative_path.as_posix()!r}."
        )

    event_class = int(relative_path.parent.name)
    filename = relative_path.name
    real_match = REAL_INSTANCE.match(filename)

    if real_match:
        source = "REAL"
        well, calendar_day = real_match.groups()
        group = f"real:{well}:{calendar_day}"
    elif filename.startswith("SIMULATED_"):
        source = "SIMULATED"
        group = f"simulated:{relative_path.as_posix()}"
    elif filename.startswith("DRAWN_"):
        source = "DRAWN"
        group = f"drawn:{relative_path.as_posix()}"
    else:
        raise ValueError(f"Unrecognized 3W instance name: {filename!r}.")

    return InstanceRecord(
        path=relative_path.as_posix(),
        event_class=event_class,
        source=source,
        group=group,
    )


def discover_instances(dataset_root: Path) -> list[InstanceRecord]:
    """Discover canonical Parquet instances below a dataset root."""

    dataset_root = dataset_root.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    paths = sorted(
        path.relative_to(dataset_root) for path in dataset_root.glob("[0-9]/*.parquet")
    )
    if not paths:
        raise FileNotFoundError(f"No '<class>/*.parquet' files found in {dataset_root}")
    return [_record(path) for path in paths]


def assign_validation_splits(
    records: list[InstanceRecord], seed: int = DEFAULT_SEED
) -> dict[str, int]:
    """Assign every instance to one leakage-aware validation split."""

    if len({record.path for record in records}) != len(records):
        raise ValueError("Instance paths must be unique.")

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=seed,
    )
    strata = [record.stratum for record in records]
    groups = [record.group for record in records]
    assignment: dict[str, int] = {}

    # A few class/source strata contain fewer than five real groups. Overall
    # class completeness is checked below, while source stratification remains
    # best-effort for those rare cells.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The least populated class in y has only .*",
            category=UserWarning,
        )
        split_indices = list(splitter.split(records, strata, groups))

    for split_index, (_, validation_indices) in enumerate(split_indices, start=1):
        for index in validation_indices:
            assignment[records[index].path] = split_index

    if len(assignment) != len(records):
        raise RuntimeError("Every instance must be assigned to one validation split.")

    group_splits: dict[str, set[int]] = defaultdict(set)
    class_splits: dict[int, set[int]] = defaultdict(set)
    for record in records:
        split = assignment[record.path]
        group_splits[record.group].add(split)
        class_splits[record.event_class].add(split)

    leaking_groups = [
        group for group, splits in group_splits.items() if len(splits) > 1
    ]
    if leaking_groups:
        raise RuntimeError(f"Leakage groups crossed splits: {leaking_groups[:5]}")

    expected_splits = set(range(1, N_SPLITS + 1))
    incomplete_classes = {
        event_class: sorted(expected_splits - splits)
        for event_class, splits in class_splits.items()
        if splits != expected_splits
    }
    if incomplete_classes:
        raise RuntimeError(
            "Every event class must occur in every validation split; missing "
            f"{incomplete_classes}."
        )

    return assignment


def write_matrix(
    records: list[InstanceRecord], assignment: dict[str, int], matrix_path: Path
) -> None:
    """Write the canonical instance-by-split binary matrix."""

    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    with matrix_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=("instance", *SPLIT_COLUMNS),
            lineterminator="\n",
        )
        writer.writeheader()
        for record in sorted(records, key=lambda item: item.path):
            validation_split = assignment[record.path]
            row: dict[str, str | int] = {"instance": record.path}
            row.update(
                {
                    column: int(number == validation_split)
                    for number, column in enumerate(SPLIT_COLUMNS, start=1)
                }
            )
            writer.writerow(row)


def _summary(records: list[InstanceRecord], assignment: dict[str, int]) -> list[str]:
    """Build concise class/source balance lines for organizer review."""

    counts = Counter(
        (assignment[record.path], record.event_class, record.source)
        for record in records
    )
    lines = []
    for split in range(1, N_SPLITS + 1):
        total = sum(count for key, count in counts.items() if key[0] == split)
        class_counts = Counter()
        for (candidate_split, event_class, _source), count in counts.items():
            if candidate_split == split:
                class_counts[event_class] += count
        lines.append(
            f"split {split}: {total} validation instances; "
            + ", ".join(
                f"class {event_class}={class_counts[event_class]}"
                for event_class in sorted(class_counts)
            )
        )
    return lines


def parse_args() -> argparse.Namespace:
    """Parse organizer overrides after loading repository configuration."""

    load_env_file()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=configured_path("THREE_W_DATASET_PATH", REPO_ROOT / "dataset"),
        help="Complete 3W Dataset root (default: THREE_W_DATASET_PATH).",
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=configured_path(
            "THREE_W_WORKSHOP_MATRIX_PATH", WORKSHOP_DIR / "instance_split_matrix.csv"
        ),
        help="Matrix output path (default: THREE_W_WORKSHOP_MATRIX_PATH).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=configured_int("THREE_W_WORKSHOP_RANDOM_SEED", DEFAULT_SEED),
        help="Grouped stratification seed (default: THREE_W_WORKSHOP_RANDOM_SEED).",
    )
    return parser.parse_args()


def main() -> None:
    """Generate the organizer-owned matrix and stop before conversion."""

    args = parse_args()
    records = discover_instances(args.dataset_root)
    assignment = assign_validation_splits(records, seed=args.seed)
    write_matrix(records, assignment, args.matrix)
    print(f"Wrote {len(records)} rows to {args.matrix}")
    for line in _summary(records, assignment):
        print(line)


if __name__ == "__main__":
    main()
