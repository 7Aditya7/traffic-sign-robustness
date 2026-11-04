"""Lazy Hugging Face model loading and inference helpers."""

from collections.abc import Iterable
from typing import Any

from .config import MODEL_ID


def select_device(preferred: str = "auto") -> Any:
    """Return a torch.device, importing torch only when explicitly requested."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for model inference.") from exc
    if preferred not in {"auto", "cpu", "cuda"}:
        raise ValueError("preferred must be 'auto', 'cpu', or 'cuda'.")
    if preferred == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available.")
    return torch.device("cuda" if preferred == "cuda" or (preferred == "auto" and torch.cuda.is_available()) else "cpu")


def load_model(model_id: str = MODEL_ID, device: str = "auto") -> tuple[Any, Any, Any]:
    """Download and load the processor/model only when this function is called."""
    try:
        from transformers import AutoImageProcessor, AutoModelForImageClassification
    except ImportError as exc:
        raise RuntimeError("transformers is required to load the model.") from exc
    target = select_device(device)
    try:
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForImageClassification.from_pretrained(model_id)
        model.to(target)
        model.eval()
    except Exception as exc:
        raise RuntimeError(f"Unable to load model '{model_id}'.") from exc
    return processor, model, target


def _rgb(image: Any) -> Any:
    try:
        return image.convert("RGB")
    except AttributeError as exc:
        raise TypeError("Each image must provide a PIL-compatible convert method.") from exc


def predict_batch(processor: Any, model: Any, images: Iterable[Any], device: Any) -> Any:
    """Run a batch safely and return logits, without gradients."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for inference.") from exc
    batch = [_rgb(image) for image in images]
    if not batch:
        raise ValueError("images must contain at least one image.")
    inputs = processor(images=batch, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        return model(**inputs).logits
