from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml
from globals import CONFIG_DIR
from pydantic import BaseModel, PrivateAttr

IGNORE_STRUCTURE_KEY = "IGNORE"


class StructureNames(BaseModel, extra="forbid"):
    """
    Maps standard structure names to their alternative names.
    """

    mapping: dict[str, set[str]]
    _inverse_mapping: Optional[dict[str, str]] = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        """Build the inverse mapping: alternative -> standard name."""
        self._inverse_mapping = {value: key for key, values in self.mapping.items() for value in values if key != IGNORE_STRUCTURE_KEY}

    def __str__(self) -> str:
        """Return a formatted string of the mapping."""
        return "\n".join(f"{key}: {', '.join(values)}" for key, values in self.mapping.items())

    def __call__(self, original_name: str) -> Optional[str]:
        """Return the standardized name for the given original name, if available."""
        return self.inverse_mapping.get(original_name)

    @classmethod
    def get_path(cls) -> Path:
        """Return the file path for the structure names config."""
        return CONFIG_DIR / "structure_names.yaml"

    @property
    def inverse_mapping(self) -> Optional[dict[str, str]]:
        """Return the inverse mapping (alternative -> standard)."""
        return self._inverse_mapping

    @classmethod
    def read(cls) -> StructureNames:
        """Load the structure names mapping from a YAML file."""
        with open(cls.get_path(), "r") as stream:
            config_data = yaml.safe_load(stream)
        return cls(**config_data)

    def write(self) -> None:
        """Write the structure names mapping to a YAML file."""
        self._inverse_mapping = None  # Reset the inverse mapping to avoid serializing it.
        model_dict = {"mapping": {name: list(names) for name, names in self.mapping.items()}}
        with open(self.get_path(), "w") as file:
            yaml.dump(model_dict, file)
