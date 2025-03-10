from __future__ import annotations

from pathlib import Path

import yaml
from globals import DATA_DIR
from pydantic import BaseModel
from sklearn.model_selection import train_test_split


class DataSplit(BaseModel):
    train_ids: list[str]
    validation_ids: list[str]
    test_ids: list[str]

    def get_all_ids(self) -> list[str]:
        """Return all IDs."""
        return self.train_ids + self.validation_ids + self.test_ids

    def write(self, plan_type: str) -> None:
        """Write split to file."""
        with open(self._get_path(plan_type), "w") as file:
            yaml.dump(self.model_dump(), file)

    @classmethod
    def read(cls, plan_type: str) -> DataSplit:
        """Read split from file."""
        file_path = cls._get_path(plan_type)
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist")
        with open(file_path, "r") as stream:
            config_data = yaml.safe_load(stream)
        return cls(**config_data)

    @staticmethod
    def _get_path(plan_type: str) -> Path:
        """Get file path for the plan type."""
        return DATA_DIR / f"{plan_type}_data_split.json"


def create_data_splits(plan_ids: list[str], val_test_ratio: float, test_ratio: float) -> DataSplit:
    """Split plan IDs into train, validation, and test sets."""
    if val_test_ratio + test_ratio >= 1:
        raise ValueError("Validation and test ratio must sum to less than 1.")

    val_test_size = val_test_ratio + test_ratio
    train_ids, test_val_ids = train_test_split(plan_ids, test_size=val_test_size, random_state=42)
    test_to_validation_ratio = test_ratio / val_test_size
    val_ids, test_ids = train_test_split(test_val_ids, test_size=test_to_validation_ratio, random_state=42)
    return DataSplit(train_ids=train_ids, validation_ids=val_ids, test_ids=test_ids)
