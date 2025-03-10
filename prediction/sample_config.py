from __future__ import annotations

from typing import Any, Optional

import yaml
from case.config import CaseConfig
from globals import CONFIG_DIR
from pydantic import BaseModel, PrivateAttr, model_validator

from prediction.feature import Feature


class SampleConfig(BaseModel):
    type: str
    inputs: list[Feature]
    outputs: list[Feature]
    mandatory_features: list[Feature]
    _mandatory_data: set[tuple[str, Optional[str]]] = PrivateAttr()
    _patient_config: Optional[CaseConfig] = PrivateAttr()
    _inputs: list[str] = PrivateAttr()
    _outputs: list[str] = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        """Initialize private attributes after model creation."""
        self._patient_config = self._create_patient_config()
        self._mandatory_data = {(feature.name, None) for feature in self.mandatory_features}
        self._mandatory_data.update(
            (feature.name, sub_feature) for feature in self.mandatory_features for sub_feature in (feature.sub_features or [])
        )
        self._inputs = self._stringify_feature_names(self.inputs)
        self._outputs = self._stringify_feature_names(self.outputs)

    @staticmethod
    def _stringify_feature_names(features: list[Feature]) -> list[str]:
        """Return feature names (including sub-features) as strings."""
        names = []
        for feature in features:
            if feature.sub_features:
                names.extend(feature.get_complete_name(sub) for sub in feature.sub_features)
            else:
                names.append(feature.name)
        return names

    @property
    def input_names(self) -> list[str]:
        """List of input feature names."""
        return self._inputs

    @property
    def output_names(self) -> list[str]:
        """List of output feature names."""
        return self._outputs

    def is_mandatory(self, feature_name: str, sub_feature: Optional[str] = None) -> bool:
        """Return True if the feature (and optional sub-feature) is mandatory."""
        return (feature_name, sub_feature) in self._mandatory_data if sub_feature else (feature_name, None) in self._mandatory_data

    def get_structure_names(self, only_mandatory: bool = False) -> set[str]:
        """Return structure names from features containing 'structure'."""
        structure_names: list[str] = []
        features_to_check = self.mandatory_features if only_mandatory else (*self.inputs, *self.outputs)
        for feature in features_to_check:
            if "structure" in feature.name.lower() and feature.sub_features:
                structure_names.extend(feature.sub_features)
        return set(structure_names)

    @classmethod
    def read(cls, config_name: str) -> SampleConfig:
        """Load config from a YAML file."""
        path = CONFIG_DIR / f"{config_name}.yaml"
        with open(path, "r") as stream:
            config_data = yaml.safe_load(stream)
        return cls(**config_data)

    @property
    def patient_config(self) -> CaseConfig:
        """Return the corresponding CaseConfig."""
        return self._patient_config

    def _create_patient_config(self) -> CaseConfig:
        """Build a CaseConfig based on inputs and outputs."""
        patient_config = {"type": self.type}
        for feature in (*self.inputs, *self.outputs):
            patient_config[feature.name] = [sub for sub in feature.sub_features] if feature.sub_features else True
        return CaseConfig(**patient_config)

    @staticmethod
    def _collect_feature_names(feature_list: list[Feature]) -> set[tuple[str, Optional[str]]]:
        """Collect feature names as (name, sub_feature) tuples."""
        names = set()
        for feature in feature_list:
            names.add((feature.name, None))
            if feature.sub_features:
                for sub in feature.sub_features:
                    names.add((feature.name, sub))
        return names

    @model_validator(mode="after")
    def validate_config(self) -> SampleConfig:
        """Check that mandatory features are defined and no feature is both input and output."""
        input_names = self._collect_feature_names(self.inputs)
        output_names = self._collect_feature_names(self.outputs)
        mandatory_names = self._collect_feature_names(self.mandatory_features)

        if missing := mandatory_names.difference({*input_names, *output_names}):
            raise ValueError(f"Mandatory features missing: {missing}")
        if overlap := output_names.intersection(input_names):
            raise ValueError(f"Features defined as both input and output: {overlap}")
        return self
