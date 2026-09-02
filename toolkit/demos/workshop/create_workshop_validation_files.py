"""Create deterministic 3W Toolkit validation lists from the fixed split CSV.

This is the challenge-facing script. It never assigns or shuffles instances;
it only validates the distributed matrix and converts each split column to a
sorted, one-dataset-relative-path-per-line text file.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

N_SPLITS = 5
SPLIT_COLUMNS = tuple(f"split_{number}" for number in range(1, N_SPLITS + 1))
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


def read_matrix(matrix_path: Path) -> list[dict[str, str]]:
    """Read and validate the challenge's fixed five-split matrix."""

    with matrix_path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        expected_columns = ["instance", *SPLIT_COLUMNS]
        if reader.fieldnames != expected_columns:
            raise ValueError(
                f"Expected matrix columns {expected_columns}, got {reader.fieldnames}."
            )
        rows = list(reader)

    if not rows:
        raise ValueError("The split matrix must contain at least one instance.")

    instances = [row["instance"] for row in rows]
    if any(not instance for instance in instances):
        raise ValueError("Every matrix row must identify an instance path.")
    if len(set(instances)) != len(instances):
        raise ValueError("The split matrix contains duplicate instance paths.")

    for row in rows:
        indicators = [row[column] for column in SPLIT_COLUMNS]
        if any(value not in {"0", "1"} for value in indicators):
            raise ValueError(
                f"Split indicators for {row['instance']!r} must be binary."
            )
        if sum(map(int, indicators)) != 1:
            raise ValueError(
                f"Instance {row['instance']!r} must be validation exactly once."
            )
    return rows


def matrix_to_validation_files(matrix_path: Path, output_dir: Path) -> list[Path]:
    """Convert the matrix to five deterministic Toolkit-compatible lists."""

    rows = read_matrix(matrix_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    for split_number, column in enumerate(SPLIT_COLUMNS, start=1):
        instances = sorted(row["instance"] for row in rows if row[column] == "1")
        path = output_dir / f"validation_split_{split_number}.txt"
        path.write_text(
            "".join(f"{instance}\n" for instance in instances), encoding="utf-8"
        )
        outputs.append(path)

    return outputs


def parse_args() -> argparse.Namespace:
    """Parse deterministic conversion paths from .env or CLI overrides."""

    load_env_file()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        type=Path,
        default=configured_path(
            "THREE_W_WORKSHOP_MATRIX_PATH", WORKSHOP_DIR / "instance_split_matrix.csv"
        ),
        help="Fixed matrix input path (default: THREE_W_WORKSHOP_MATRIX_PATH).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=configured_path("THREE_W_WORKSHOP_SPLIT_DIR", WORKSHOP_DIR / "splits"),
        help="Validation-list directory (default: THREE_W_WORKSHOP_SPLIT_DIR).",
    )
    return parser.parse_args()


def main() -> None:
    """Validate the fixed matrix and write its five text representations."""

    args = parse_args()
    outputs = matrix_to_validation_files(args.matrix, args.output_dir)
    print(f"Validated {len(read_matrix(args.matrix))} matrix rows.")
    print("Wrote validation lists:")
    for output in outputs:
        print(f"- {output}")


if __name__ == "__main__":
    main()
