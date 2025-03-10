from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from case import Case
from numpy.typing import NDArray
from utils import sitk_image_to_array

from prediction.feature import Feature
from prediction.sample_config import SampleConfig

logger = logging.getLogger(__name__)


@dataclass
class Sample:
    """
    A sample representing case data with extracted input and output features.
    """

    data: dict[str, str | NDArray]

    @classmethod
    def from_patient(cls, patient: Case, prediction_config: SampleConfig) -> Sample:
        """Create a Sample from a patient using the given prediction configuration."""
        input_data = _get_patient_attributes_as_dict(patient, prediction_config.inputs)
        output_data = _get_patient_attributes_as_dict(patient, prediction_config.outputs)
        return Sample({"id": patient.id, **input_data, **output_data})

    def __repr__(self) -> str:
        return f"Sample({self.data['id']})"

    def __getitem__(self, key: str) -> str | NDArray:
        return self.data[key]

    def get_image_features(self):
        """Yield (name, data) pairs for image features in the sample."""
        for name, data in self.data.items():
            if isinstance(data, np.ndarray):
                yield name, data


def _get_patient_attributes_as_dict(patient: Case, features: list[Feature]) -> dict[str, NDArray]:
    """
    Extract features from a patient and return a dictionary mapping feature names to image data.

    For features with sub-features, it extracts each sub-feature image; otherwise, it extracts the
    image from the patient's attribute. If an attribute is missing, a placeholder image is used.
    """
    feature_dict = {}
    for feature in features:
        if feature.sub_features:
            contents = getattr(patient, feature.name)
            if not isinstance(contents, dict):
                continue
            for sub_feature in feature.sub_features:
                name = feature.get_complete_name(sub_feature)
                image = contents[sub_feature].image if sub_feature in contents else patient.get_placeholder_image()
                feature_dict[name] = np.expand_dims(sitk_image_to_array(image), axis=0)
        else:
            attr = getattr(patient, feature.name, None)
            if attr is not None and hasattr(attr, "image"):
                image = attr.image
            else:
                image = patient.get_placeholder_image()
            feature_dict[feature.name] = np.expand_dims(sitk_image_to_array(image), axis=0)
    return feature_dict
