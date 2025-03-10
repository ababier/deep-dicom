from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk


@dataclass(kw_only=True)
class Dose:
    image: sitk.Image

    def get_dose_vector(self) -> np.ndarray:
        # Convert the dose image to a numpy array.
        return sitk.GetArrayFromImage(self.image)
