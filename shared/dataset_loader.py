"""Lazy inspection and loading helpers for the Hugging Face GTSRB dataset."""

from typing import Any

from .config import DATASET_ID


def _datasets() -> Any:
    try:
        from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset
    except ImportError as exc:
        raise RuntimeError("The datasets package is required for dataset operations.") from exc
    return get_dataset_config_names, get_dataset_split_names, load_dataset


def inspect_dataset(dataset_id: str = DATASET_ID) -> dict[str, Any]:
    """Inspect configurations and splits; this may access Hugging Face metadata."""
    get_configs, get_splits, _ = _datasets()
    configurations = get_configs(dataset_id)
    return {
        "dataset_id": dataset_id,
        "configurations": configurations,
        "splits": {config: get_splits(dataset_id, config) for config in configurations},
    }


def load_split(dataset_id: str = DATASET_ID, split: str = "train", config: str | None = None) -> Any:
    """Load a requested split only when explicitly called."""
    _, _, load_dataset = _datasets()
    try:
        return load_dataset(dataset_id, name=config, split=split)
    except Exception as exc:
        detail = f"config={config!r}, split={split!r}"
        raise RuntimeError(f"Unable to load dataset '{dataset_id}' ({detail}).") from exc


def summarize_dataset(dataset: Any) -> dict[str, Any]:
    """Return basic metadata from an already loaded dataset without downloading."""
    features = getattr(dataset, "features", None)
    label_feature = None
    if features:
        for name in ("label", "labels", "fine_label"):
            candidate = features.get(name)
            if candidate is not None:
                label_feature = candidate
                break
    return {
        "num_rows": len(dataset),
        "features": list(features.keys()) if features else [],
        "label_names": list(getattr(label_feature, "names", []) or []),
    }
