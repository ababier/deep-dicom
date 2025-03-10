from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, floor
from pathlib import Path

import numpy as np
import yaml
from globals import CONFIG_DIR
from numpy.typing import NDArray


@dataclass
class ScalingParams:
    scale_factor: int = 1
    offset: int = 0


@dataclass
class ScalingConfig:
    scale_parameters: dict[str, ScalingParams]

    @classmethod
    def from_feature_names(cls, features: list[str]) -> ScalingConfig:
        """
        Initialize scaling configuration for the given features.

        Each feature is assigned default scaling parameters.
        """
        return ScalingConfig(scale_parameters={feature: ScalingParams() for feature in features})

    def update(self, feature_name: str, data: NDArray) -> None:
        """
        Update scaling parameters for a feature based on its data.

        The scale factor is set to the difference between the max and min values
        (if larger than the current factor), and the offset is updated to the minimum value.
        """
        if feature_name not in self.scale_parameters:
            raise ValueError(f"Feature '{feature_name}' not found in scale configuration.")
        if isinstance(data, np.ndarray):
            scale_params = self.scale_parameters[feature_name]
            scale_params.scale_factor = max(ceil(data.max()) - floor(data.min()), scale_params.scale_factor)
            scale_params.offset = min(floor(data.min()), scale_params.offset)

    def __getitem__(self, key: str) -> ScalingParams:
        """
        Retrieve scaling parameters for a given feature.
        """
        return self.scale_parameters[key] if key in self.scale_parameters else None

    @classmethod
    def get_path(cls) -> Path:
        """Return the file path for the scaling configuration."""
        return CONFIG_DIR / "scaling_factors.yaml"

    @classmethod
    def read(cls) -> ScalingConfig:
        """
        Read the scaling configuration from a YAML file.
        """
        config_path = cls.get_path()
        if not config_path.exists():
            raise FileNotFoundError(f"Scale config file '{config_path}' not found.")
        with open(config_path, "r") as file:
            data = yaml.safe_load(file)
        return cls(scale_parameters={key: ScalingParams(**value) for key, value in data.items()})

    def write(self) -> None:
        """Write the current scaling configuration to a YAML file."""
        with open(self.get_path(), "w") as file:
            yaml.dump(asdict(self)["scale_parameters"], file, default_flow_style=False)

    def add_buffer(self, offset: int = 0, scale: int = 0) -> None:
        """
        Remove trivial features and add a buffer to scaling parameters.

        Features with default values (scale_factor == 1 and offset == 0) are removed.
        Then, the specified scale is added and the offset is subtracted from each parameter.
        """
        trivial_features = {name for name, params in self.scale_parameters.items() if params.scale_factor == 1 and params.offset == 0}
        self.scale_parameters = {name: scale_params for name, scale_params in self.scale_parameters.items() if name not in trivial_features}
        for scale_params in self.scale_parameters.values():
            scale_params.scale_factor = ceil(scale_params.scale_factor + scale)
            scale_params.offset = floor(scale_params.offset - offset)
