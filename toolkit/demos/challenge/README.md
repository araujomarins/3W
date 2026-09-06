# 3W challenge baseline examples

This directory contains the fixed validation artifacts and two reference
notebooks for the 3W modeling challenge:

- **multiclass sample classification:** predict labels `0` through `9`;
- **fault detection:** predict whether a sample is normal (`0`) or faulty (`1`).

The notebooks are deliberately simple baselines. They show the complete data
and evaluation workflow with a default model and no hyperparameter search.
Participants can improve the modeling approach, but comparable results must use
the same dataset version, cleaning rules, and fixed validation files.

## Configure the challenge

Create a virtual environment, install the project with its development tools,
and copy the example configuration:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env
```

If you already use a Conda environment, activate it and run
`python -m pip install -e ".[dev]"` instead. The development dependencies
include the Jupyter Notebook application.

The scripts and notebooks use `python-dotenv` to load `.env` automatically.
Update these values before running them:

- `THREE_W_REPOSITORY_PATH`: absolute path to the local 3W repository;
- `THREE_W_DATASET_PATH`: absolute path to the complete 3W Dataset v2.0.0;
- `THREE_W_CHALLENGE_PATH`: directory containing these challenge materials;
- `THREE_W_CHALLENGE_MATRIX_PATH`: fixed instance split matrix;
- `THREE_W_CHALLENGE_SPLIT_DIR`: directory for the five validation text files;
- `THREE_W_CHALLENGE_RANDOM_SEED`: seed used by the reproducible baseline model.

The last four values in `.env.example` reuse the repository and challenge paths
with `${VARIABLE}` expansion, so most users only need to replace the repository
and dataset paths.

## Fixed evaluation protocol

The checked-in matrix covers all **2,228 instance files** in the complete 3W
Dataset v2.0.0. Every instance is assigned to validation in exactly one of five
runs and is used for training in the other four. The split is made at instance
level, which prevents observations from the same Parquet file from appearing in
both training and validation in one run.

Both baseline notebooks apply the same rules before modeling:

1. load every instance listed across the five fixed validation files;
2. remove only the consecutive rows with missing `class` labels at the beginning
   of each instance, before its first observed label;
3. record which remaining labels were originally observed, then fill later gaps
   within each instance with `FillLabelsConfig(fill_method="nearest")`;
4. map transient labels to their steady class with the Toolkit-configured
   offset (`105` becomes `5`);
5. keep only the resulting labels `0` through `9`;
6. for each run, fit the 3W Toolkit's `CleanSignals` thresholds on that run's
   training instances only and apply them unchanged to validation;
7. fit Toolkit mean imputation on that run's cleaned training observations
   only and apply the fitted values unchanged to validation;
8. include filled labels during training, but exclude them from **every validation
   metric**, including ROC-AUC and confusion matrices;
9. do not apply dataset-wide normalization or use validation observations to
   estimate signal-preprocessing statistics.

The Toolkit's [FillLabels implementation](../../ThreeWToolkit/preprocessing/fill_labels.py)
uses nearest interpolation by row position, followed by backward/forward fill
for any remaining gaps. It does not fit a classifier or use another instance's
labels. Trailing gaps are filled from the last known label. An entirely unlabeled
instance is skipped and counted because it has no known label to interpolate.

The shared `prepare_instance_labels` function in `challenge_helpers.py` keeps an
original-label mask in metadata, outside the model's features. Both the binary
and multiclass evaluation functions apply this mask to targets and predictions
before calculating any metric. Validation rows with filled labels can pass through
preprocessing and prediction, but they never become evaluation ground truth.

Every eligible observation is loaded. The notebooks use `TRAINING_SAMPLE_STRIDE = 100`, which
trains the model on every 100th observation within each training instance to
keep the five-fold example practical. Set it to `1` to use every training
observation, or to another `N` to use every Nth training observation.
Fold-specific preprocessing still uses the complete retained training fold,
including rows with filled labels. Validation is never subsampled: all retained
rows are predicted and all originally labeled eligible rows are scored. Report
the chosen stride when comparing results. The saved notebook outputs follow
this label policy and use the training stride shown in the code.

For fault detection, cleaned label `0` remains normal and labels `1` through
`9` become faulty. For multiclass classification, labels `0` through `9` are
used directly.

## Create the validation files

`create_challenge_validation_files.py` is provided to save participants the
time of converting the fixed CSV matrix by hand. The 3W Toolkit accepts a text
file containing one dataset-relative instance path per line through
`ParquetDatasetConfig(split="list", file_list=...)`. The script validates the
matrix and writes exactly that format for each of the five runs; it does not
shuffle or reassign any instance.

From the repository root, run:

```bash
python toolkit/demos/challenge/create_challenge_validation_files.py
```

The files are written to `THREE_W_CHALLENGE_SPLIT_DIR` and named
`validation_split_1.txt` through `validation_split_5.txt`.

## Notebook examples

- `multiclass_sample_classification.ipynb` demonstrates the ten-label track;
- `fault_detection.ipynb` demonstrates the binary normal-versus-faulty track.

Each notebook loads the fixed validation files with the 3W Toolkit, fits a
fresh default decision tree for each run, and reports per-run metrics plus an
aggregated confusion matrix. These results are a reproducible starting point,
not a target model architecture for participants. During each run, the notebook
prints the training and validation sizes and progress through fold-specific
cleaning and imputation, training, prediction, and metric calculation.

`challenge_helpers.py` shares the label policy and data preparation between the
notebooks. The training loop calls `split_fold_data`, `prepare_fold_data`,
`train_default_model`, and the notebook's `evaluate_validation` function in order.
The loading summary counts removed leading rows, entirely unlabeled instances,
unsupported labels, and retained filled labels. Fold reports distinguish training
rows with filled labels, all predicted validation rows (`validation_rows`),
originally labeled rows scored (`validation_samples`), and filled validation
labels excluded (`validation_imputed_labels_excluded`).

Both notebooks report accuracy, balanced accuracy, macro-F1, and a pooled
row-normalized confusion matrix. Fault detection also reports faulty-class
precision, recall, F1, and ROC-AUC from fault probabilities. Scalar metrics are
reported per fold and as an arithmetic mean. ROC-AUC is `NaN` when the scored
validation labels contain only one class, and is excluded from that mean.

## Run the notebooks

After configuring `.env`, activate your environment and launch Jupyter from the
repository root:

```bash
jupyter notebook toolkit/demos/challenge
```

Open a notebook and choose **Kernel → Restart Kernel and Run All Cells**. A full
run reads all instances from the complete dataset, so runtime and
memory use will depend on the machine.
