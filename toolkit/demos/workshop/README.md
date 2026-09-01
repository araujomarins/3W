# Workshop modeling examples

This folder contains two deliberately simple sample-level modeling tutorials:

- `multiclass_sample_classification.ipynb`: predict labels 0 through 9;
- `fault_detection.ipynb`: predict normal (`0`) versus faulty (`1`).

Both notebooks use the same fixed five-split instance partition and a 3W
Toolkit model with default model hyperparameters. They do not perform model
selection or hyperparameter search.

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

## Rebuild the split artifacts

To regenerate the matrix from a complete v2.0.0 dataset and then recreate the
five text lists:

```bash
python toolkit/demos/workshop/prepare_workshop_splits.py \
  --dataset-root /path/to/3W/dataset
```

To validate the checked-in matrix and recreate only the text files:

```bash
python toolkit/demos/workshop/prepare_workshop_splits.py
```

Set `THREE_W_DATASET_PATH` before running a notebook if the complete dataset is
not in this repository's `dataset/` directory. For a quick workshop smoke run,
set `THREE_W_WORKSHOP_MAX_INSTANCES_PER_CLASS=10`; leave it unset or set it to
`0` to use all instances.
