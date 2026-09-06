"""Shared data preparation for the two challenge notebooks.

Originally observed labels are tracked in metadata, never in model features.
The Toolkit handles label filling, signal cleaning, imputation, and training.
"""

from collections import Counter
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from tqdm.auto import tqdm

from ThreeWToolkit.core.base_dataset import BaseDataset
from ThreeWToolkit.core.dataset_outputs import DatasetOutputs
from ThreeWToolkit.dataset import (
    ParquetDataset,
    ParquetDatasetConfig,
    SubsetDataset,
    TransformConfig,
    TransformedDataset,
)
from ThreeWToolkit.models import SklearnModelsConfig
from ThreeWToolkit.preprocessing import (
    CleanSignalsConfig,
    FillLabelsConfig,
    ImputeMissingConfig,
    SequentialPreprocessingAdapterConfig,
)
from ThreeWToolkit.trainer import SklearnTrainerConfig
from ThreeWToolkit.utils.data_utils import get_config_dataset_ini


class ChallengeDataset(BaseDataset):
    """In-memory instances accepted by Toolkit preprocessors and trainers."""

    def __init__(self, events: list[DatasetOutputs]):
        self.events = events

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, index: int) -> DatasetOutputs:
        return self.events[index]


def observed_label_mask(event: DatasetOutputs) -> np.ndarray:
    """Require one provenance flag per row; never infer it from filled labels."""
    mask = np.asarray(event.metadata.get("label_observed"))
    if mask.dtype != np.bool_ or mask.shape != (len(event.signal),):
        raise ValueError("Missing or misaligned original-label mask.")
    return mask


def prepare_instance_labels(
    event: DatasetOutputs, *, binary: bool, transient_offset: int
) -> tuple[DatasetOutputs | None, dict[str, int]]:
    """Trim the unlabeled prefix, fill later gaps, and retain label provenance.

    Nearest interpolation uses row positions within this instance. The Toolkit
    also fills trailing gaps from the last known label. An entirely unlabeled
    instance is skipped because it provides no label from which to interpolate.
    """
    if event.label is None:
        raise ValueError("The configured target column did not produce labels.")
    raw_labels = event.label.reset_index(drop=True)
    known_positions = np.flatnonzero(raw_labels.notna().to_numpy())
    start = int(known_positions[0]) if len(known_positions) else len(raw_labels)
    counts = {
        "leading_unlabeled_removed": start,
        "entirely_unlabeled_instances": int(not len(known_positions)),
        "unsupported_labels_removed": 0,
        "imputed_labels_retained": 0,
    }
    if not len(known_positions):
        return None, counts

    trimmed_labels = raw_labels.iloc[start:].reset_index(drop=True)
    observed = trimmed_labels.notna().to_numpy()
    trimmed = DatasetOutputs(
        signal=event.signal.iloc[start:].reset_index(drop=True),
        label=trimmed_labels.astype(float),
        metadata=event.metadata,
    )
    filled = FillLabelsConfig(fill_method="nearest").build().transform(trimmed)
    if filled.label is None or filled.label.isna().any():
        raise ValueError("Toolkit label filling left missing labels.")

    # Fill before mapping: transient and established labels remain valid anchors.
    labels = filled.label.astype("int64").to_numpy() % transient_offset
    allowed = np.isin(labels, np.arange(10))
    counts["unsupported_labels_removed"] = int((~allowed).sum())
    if not allowed.any():
        return None, counts
    positions = np.flatnonzero(allowed)
    observed = observed[allowed]
    labels = labels[allowed]
    counts["imputed_labels_retained"] = int((~observed).sum())
    target = (labels != 0).astype("int8") if binary else labels
    prepared = DatasetOutputs(
        signal=filled.signal.iloc[positions].reset_index(drop=True),
        label=pd.Series(target, name="target"),
        metadata={
            **event.metadata,
            "file_name": Path(event.metadata["file_name"]).as_posix(),
            "label_observed": observed,
        },
    )
    return prepared, counts


def load_challenge_samples(
    dataset_path: Path, files: list[str], *, binary: bool
) -> tuple[ChallengeDataset, dict[str, int]]:
    """Load the supplied instances and prepare their labels without sampling."""
    config = get_config_dataset_ini()
    columns = cast(list[str], config["COLUMNS_DATA_FILES"])
    transient_offset = cast(int, config["TRANSIENT_OFFSET"])
    feature_columns = [
        column for column in columns if column not in {"timestamp", "class", "state"}
    ]
    dataset = ParquetDataset(
        ParquetDatasetConfig(
            path=dataset_path,
            split="list",
            file_list=files,
            columns=feature_columns,
            target_column="class",
        )
    )
    events = []
    totals: Counter[str] = Counter()
    for event in tqdm(dataset, total=len(dataset), unit="instance"):
        prepared, counts = prepare_instance_labels(
            event, binary=binary, transient_offset=transient_offset
        )
        totals.update(counts)
        if prepared is not None:
            events.append(prepared)
    if not events:
        raise RuntimeError("No eligible samples were loaded.")
    return ChallengeDataset(events), dict(totals)


def split_fold_data(samples: BaseDataset, validation_instances: set[str]):
    """Keep complete instances on the fixed training or validation side."""
    train_indices: list[int] = []
    validation_indices: list[int] = []
    for index, event in enumerate(samples):
        indices = (
            validation_indices
            if str(event.metadata["file_name"]) in validation_instances
            else train_indices
        )
        indices.append(index)
    if not train_indices or not validation_indices:
        raise ValueError("Every fold needs eligible training and validation instances.")
    return SubsetDataset(samples, train_indices), SubsetDataset(
        samples, validation_indices
    )


def stride_training_rows(dataset: BaseDataset, stride: int) -> BaseDataset:
    """Select every Nth training row, keeping labels and provenance aligned."""
    if not isinstance(stride, int) or isinstance(stride, bool) or stride < 1:
        raise ValueError("Training sample stride must be a positive integer.")
    if stride == 1:
        return dataset

    def select_rows(event: DatasetOutputs) -> DatasetOutputs:
        positions = np.arange(0, len(event.signal), stride)
        if event.label is None:
            raise ValueError("Training labels are required.")
        return DatasetOutputs(
            signal=event.signal.iloc[positions].reset_index(drop=True),
            label=event.label.iloc[positions].reset_index(drop=True),
            metadata={
                **event.metadata,
                "label_observed": observed_label_mask(event)[positions],
            },
        )

    return TransformedDataset(dataset, select_rows)


def prepare_fold_data(raw_train: BaseDataset, raw_validation: BaseDataset, stride: int):
    """Fit signal processing on the full training fold, then apply training stride.

    Toolkit CleanSignals and mean imputation preserve row order and metadata,
    so the validation mask follows the same order as trainer predictions.
    """
    # Validate the stride before fitting preprocessing or calculating counts.
    selected_train = stride_training_rows(raw_train, stride)
    validation_mask = np.concatenate(
        [observed_label_mask(event) for event in raw_validation]
    )
    if not validation_mask.any():
        raise ValueError("No original validation labels are available for scoring.")
    counts = {
        "available_train_samples": sum(len(event.signal) for event in raw_train),
        "train_samples": sum(len(event.signal) for event in selected_train),
        "train_imputed_labels": sum(
            int((~observed_label_mask(event)).sum()) for event in selected_train
        ),
        "validation_rows": len(validation_mask),
        "validation_samples": int(validation_mask.sum()),
        "validation_imputed_labels_excluded": int((~validation_mask).sum()),
    }
    print(
        f"  Training rows: {counts['train_samples']:,} of "
        f"{counts['available_train_samples']:,} available (stride={stride}); "
        f"{counts['train_imputed_labels']:,} have filled labels.",
        flush=True,
    )
    print(
        f"  Validation rows: {counts['validation_rows']:,}; "
        f"scoring {counts['validation_samples']:,} original labels; "
        f"excluding {counts['validation_imputed_labels_excluded']:,} filled labels.",
        flush=True,
    )
    print(
        "  Fitting Toolkit signal cleaning and mean imputation on training...",
        flush=True,
    )
    preprocessor = TransformConfig(
        pre_processing=SequentialPreprocessingAdapterConfig(
            steps=[CleanSignalsConfig(), ImputeMissingConfig(strategy="mean")]
        )
    ).build()
    preprocessor.fit(raw_train)
    train = stride_training_rows(preprocessor.transform(raw_train), stride)
    validation = preprocessor.transform(raw_validation)
    return train, validation, validation_mask, counts


def train_default_model(train: BaseDataset, random_seed: int):
    """Fit a fresh Toolkit decision tree with default model hyperparameters."""
    print("  Training the decision tree...", flush=True)
    trainer = SklearnTrainerConfig(
        config_model=SklearnModelsConfig(
            model_type=DecisionTreeClassifier,
            model_params={"random_state": random_seed},
        ),
        seed=random_seed,
    ).build()
    trainer.train(train)
    return trainer


def select_observed_predictions(prediction_result, validation_mask: np.ndarray):
    """Apply the original-label mask to both targets and model outputs."""
    y_true = prediction_result.y_true
    y_pred = prediction_result.y_pred
    if y_true is None:
        raise ValueError("Validation labels are required for evaluation.")
    if (
        validation_mask.dtype != np.bool_
        or validation_mask.shape != (len(y_true),)
        or len(y_pred) != len(y_true)
    ):
        raise ValueError(
            "Validation predictions and original-label mask are misaligned."
        )
    if not validation_mask.any():
        raise ValueError("No original validation labels are available for scoring.")
    return y_true[validation_mask], y_pred[validation_mask]
