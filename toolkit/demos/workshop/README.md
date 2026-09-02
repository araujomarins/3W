# Workshop modeling examples

This folder contains two deliberately simple sample-level modeling tutorials:

- `multiclass_sample_classification.ipynb`: predict labels 0 through 9;
- `fault_detection.ipynb`: predict normal (`0`) versus faulty (`1`).

Both notebooks use the same fixed five-split instance partition and a 3W
Toolkit model with default model hyperparameters. They do not perform model
selection or hyperparameter search.

## Configure the workshop

All workshop entry points automatically read `.env` from the repository root.
Create it once from the documented template:

```bash
cp .env.example .env
```

Open `.env` and replace `THREE_W_DATASET_PATH` with the absolute path to your
complete 3W Dataset v2.0.0 directory. Its immediate children must be the event
class directories (`0/` through `9/`) containing the Parquet instances.

The remaining values have working repository-relative defaults:

- `THREE_W_WORKSHOP_MATRIX_PATH`: the fixed challenge CSV;
- `THREE_W_WORKSHOP_SPLIT_DIR`: where the five Toolkit text lists are written;
- `THREE_W_WORKSHOP_MAX_INSTANCES_PER_CLASS`: `0` for the complete run, or `10`
  for a quick smoke check;
- `THREE_W_WORKSHOP_SAMPLES_PER_LABEL_PER_INSTANCE`: bounded observations kept
  from each label in each instance;
- `THREE_W_WORKSHOP_RANDOM_SEED`: organizer split seed and notebook model seed.

Values already exported by the shell take precedence over `.env`. The local
`.env` is ignored by Git; `.env.example` is the shareable documentation and
must not contain machine-specific paths or secrets.

## Label rules

The notebooks apply the workshop rules before modeling:

1. observations with a missing `class` value are removed;
2. transient labels are mapped to their event class using the transient offset
   from `dataset.ini` (`105` becomes `5`, for example);
3. multiclass labels remain `0` through `9`;
4. fault-detection labels become `0` for normal and `1` for any fault.

The toolkit's signal cleanup and global normalization are then applied. The
notebooks load raw labels with `clean_data=False` because the toolkit's default
label handling fills annotation gaps, while this workshop explicitly removes
unlabeled observations.

## Five-split policy

`instance_split_matrix.csv` has one row per v2.0.0 Parquet instance. The five
indicator columns mean:

- `1`: use the instance for validation in that split;
- `0`: use the instance for training in that split.

Each instance is validation exactly once. Assignment is stratified by event
class and source. Real files from the same well and calendar day remain in the
same split, reducing leakage from closely related real events. Simulated and
drawn files are independent groups.

Whole-well grouping was intentionally not used: some real fault classes cover
only two or three wells, so five class-complete validation folds would be
impossible. Same-well/day grouping is the practical compromise for this
introductory workshop.

The five `splits/validation_split_N.txt` files contain one dataset-relative
Parquet path per line. Each list can be passed directly to
`ParquetDatasetConfig(file_list=..., split="list")`; training paths are the
complement of the selected validation list.

## Split artifacts and challenge boundary

Matrix assignment and matrix conversion are intentionally separate.

### Organizer only: generate the matrix

`generate_workshop_split_matrix.py` discovers the complete dataset, stratifies
by event class and source, keeps related real instances grouped, and writes only
`instance_split_matrix.csv`:

```bash
python toolkit/demos/workshop/generate_workshop_split_matrix.py
```

This organizer-owned script is not part of the participant challenge package.
Changing its seed or rerunning it can create a different assignment, so the
checked-in matrix is the canonical starting point distributed to everyone.

### Challenge facing: create the Toolkit lists

`create_workshop_validation_files.py` contains no assignment, shuffling, or
randomness. It validates the distributed CSV and deterministically creates the
five one-path-per-line files expected by the 3W Toolkit:

```bash
python toolkit/demos/workshop/create_workshop_validation_files.py
```

Every participant starts from the same `instance_split_matrix.csv`; the script
only converts its five indicator columns into the legacy-compatible text-list
format used by `ParquetDatasetConfig(file_list=..., split="list")`.

## Open the notebooks

After activating the `3W` environment and creating `.env`, start Jupyter from
the repository root—no environment-variable exports are needed:

```bash
jupyter notebook toolkit/demos/workshop
```

Restart the kernel and run all cells after changing `.env`.
