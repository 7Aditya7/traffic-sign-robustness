"""Explicit comparison of dataset and model traffic-sign label mappings."""

import warnings
from collections.abc import Mapping, Sequence
from typing import Any


def extract_model_mapping(config: Any) -> dict[int, str]:
    raw = getattr(config, "id2label", None)
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("Model config does not expose a usable id2label mapping.")
    return {int(key): str(value) for key, value in raw.items()}


def validate_label_mapping(dataset_names: Sequence[str], model_id2label: Mapping[int, str]) -> bool:
    """Validate IDs and semantic names; raise rather than silently guessing."""
    if not dataset_names or not model_id2label:
        raise ValueError("Both dataset names and model mappings are required.")
    expected = {index: str(name) for index, name in enumerate(dataset_names)}
    normalized_model = {int(key): str(value) for key, value in model_id2label.items()}
    if set(expected) != set(normalized_model):
        raise ValueError("Dataset and model label IDs do not cover the same classes.")
    normalized = lambda value: " ".join(value.strip().lower().split())
    if any(normalized(expected[key]) != normalized(normalized_model[key]) for key in expected):
        warnings.warn("Dataset and model label names could not be verified as identical.", UserWarning, stacklevel=2)
        return False
    return True
