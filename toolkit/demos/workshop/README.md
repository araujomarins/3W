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

- `THREE_W_WORKSHOP_MATRIX_PATH`: the fixed split CSV;
- `THREE_W_WORKSHOP_SPLIT_DIR`: where the five Toolkit text lists are written;
- `THREE_W_WORKSHOP_MAX_INSTANCES_PER_CLASS`: `0` for the complete run, or `10`
  for a quick smoke check;
- `THREE_W_WORKSHOP_SAMPLES_PER_LABEL_PER_INSTANCE`: bounded observations kept
  from each label in each instance;
- `THREE_W_WORKSHOP_RANDOM_SEED`: split-generation seed and notebook model seed.

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

## Create the Toolkit validation files

We added `create_workshop_validation_files.py` to save participants the time of
manually converting the split matrix. The 3W Toolkit already accepts a text file
containing one dataset-relative Parquet path per line through
`ParquetDatasetConfig(file_list=..., split="list")`.

The helper validates `instance_split_matrix.csv` and creates the five text files
in that format. It does not change the matrix assignment:

```bash
python toolkit/demos/workshop/create_workshop_validation_files.py
```

The generated files are written to `THREE_W_WORKSHOP_SPLIT_DIR`, which defaults
to `toolkit/demos/workshop/splits`.

## Workshop notebook examples

The workshop includes two notebooks that demonstrate how to load the generated
split files, apply the agreed label rules, train a default 3W Toolkit model, and
evaluate it over all five runs:

- `multiclass_sample_classification.ipynb`: classify samples as labels `0`
  through `9`;
- `fault_detection.ipynb`: classify samples as normal (`0`) or faulty (`1`).

## Run the workshop notebooks

After creating `.env`, activate the `3W` environment and start Jupyter from the
repository root. No environment-variable exports are needed:

```bash
conda activate 3W
jupyter notebook toolkit/demos/workshop
```

Restart the kernel and run all cells after changing `.env`.
