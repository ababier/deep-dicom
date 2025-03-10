from __future__ import annotations

from dataclasses import dataclass

import SimpleITK as sitk


@dataclass(kw_only=True)
class CT:
    """Container for a CT scan. Ensures the image is cast to sitkInt32."""

    image: sitk.Image

    def __post_init__(self) -> None:
        # Cast image to 32-bit integer format for consistency.
        self.image = sitk.Cast(self.image, sitk.sitkInt32)

    @property
    def num_slices(self) -> int:
        # Return the number of slices (depth) in the CT volume.
        return self.image.GetDepth()
