from typing import Optional

import numpy as np
from case import read_case
from numpy.typing import NDArray
from prediction.sample import Sample
from prediction.sample_config import SampleConfig
from prediction.scaling import ScalingConfig
from torch.utils.data import Dataset

from models.transformations import RotateAndFlip


class LargeImageDataset(Dataset):
    """
    Dataset for large images with optional transformations and scaling.

    Args:
        plan_config: Configuration for sample generation.
        plan_ids: List of plan IDs.
        transform: Optional transformation to apply to image data.
    """

    def __init__(self, plan_config: SampleConfig, plan_ids: list[str], transform: Optional[RotateAndFlip] = None):
        self.plan_config = plan_config
        self.plan_ids = plan_ids
        self.transform = transform
        self._scale_config = ScalingConfig.read()

    def __len__(self) -> int:
        return len(self.plan_ids)

    def __getitem__(self, idx: int) -> dict[str, NDArray | str]:
        # Read patient and create a sample.
        patient = read_case(self.plan_ids[idx], self.plan_config.patient_config)
        sample = Sample.from_patient(patient, self.plan_config).data

        # Apply optional transformations.
        if self.transform:
            self.transform.set_transform()
            for name, data in sample.items():
                if isinstance(data, np.ndarray):
                    sample[name] = self.transform(data)

        # Scale image features.
        for name, data in sample.items():
            if scale_parameters := self._scale_config[name]:
                sample[name] = (data - scale_parameters.offset) / scale_parameters.scale_factor

        return sample
