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

from dotenv import load_dotenv

N_SPLITS = 5
SPLIT_COLUMNS = tuple(f"split_{number}" for number in range(1, N_SPLITS + 1))
load_dotenv()

DEFAULT_CHALLENGE_PATH = Path(__file__).resolve().parent
REPOSITORY_PATH = Path(
    os.getenv("THREE_W_REPOSITORY_PATH", str(DEFAULT_CHALLENGE_PATH.parents[2]))
).expanduser()
CHALLENGE_PATH = Path(
    os.getenv("THREE_W_CHALLENGE_PATH", str(DEFAULT_CHALLENGE_PATH))
).expanduser()
MATRIX_PATH = Path(
    os.getenv(
        "THREE_W_CHALLENGE_MATRIX_PATH",
        str(CHALLENGE_PATH / "instance_split_matrix.csv"),
    )
).expanduser()
SPLIT_DIR = Path(
    os.getenv("THREE_W_CHALLENGE_SPLIT_DIR", str(CHALLENGE_PATH / "splits"))
).expanduser()


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

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        type=Path,
        default=MATRIX_PATH,
        help="Fixed matrix input path (default: THREE_W_CHALLENGE_MATRIX_PATH).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SPLIT_DIR,
        help="Validation-list directory (default: THREE_W_CHALLENGE_SPLIT_DIR).",
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
