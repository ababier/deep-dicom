from __future__ import annotations

from dataclasses import dataclass

import SimpleITK as sitk
from case.utils import indices_to_physical_points


@dataclass(kw_only=True)
class Mask:
    image: sitk.Image
    name: str

    @property
    def positions(self):
        # Return physical positions corresponding to non-zero indices in the mask.
        return indices_to_physical_points(self.image)

    def __add__(self, mask: Mask) -> Mask:
        # Combine two masks by summing their images and merging names if necessary.
        if self.name != mask.name:
            self.name = f"{self.name}_{mask.name}"
        self.image += mask.image
        return self

    def sample(self, base_image: sitk.Image) -> None:
        """Resample this mask to match the geometry of the base_image using nearest neighbor interpolation."""
        self.image = sitk.Resample(
            self.image, referenceImage=base_image, transform=sitk.Transform(), interpolator=sitk.sitkNearestNeighbor, defaultPixelValue=0
        )
