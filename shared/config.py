"""Central project metadata and validation without external dependencies."""

from dataclasses import dataclass

MODEL_ID = "kelvinandreas/vit-traffic-sign-GTSRB"
DATASET_ID = "tanganke/gtsrb"


@dataclass(frozen=True)
class ProjectConfig:
    model_id: str = MODEL_ID
    dataset_id: str = DATASET_ID
    python_min_version: tuple[int, int] = (3, 11)

    def validate(self) -> None:
        if not self.model_id or not self.dataset_id:
            raise ValueError("model_id and dataset_id must be non-empty.")
        if self.python_min_version < (3, 11):
            raise ValueError("Python minimum version must be 3.11 or newer.")
