from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import SimpleITK as sitk
from numpy.typing import NDArray

from case.attributes.contour import Contour
from case.attributes.ct import CT
from case.attributes.dose import Dose
from case.attributes.mask import Mask
from case.utils import resample_img


@dataclass
class Case:
    """A container for medical imaging data associated with a single patient case.

    Attributes:
        id: Unique identifier for the patient case.
        type: Type or category of the case (e.g., 'clinical', 'dose_prediction').
        ct: CT imaging data associated with the case.
        structure_contours: Dictionary mapping structure names to contour data.
        structure_masks: Dictionary mapping structure names to mask images.
        dose: Radiation dose data associated with the case.
    """

    id: str
    type: str
    ct: Optional[CT] = None
    structure_contours: Optional[dict[str, Contour]] = None
    structure_masks: Optional[dict[str, Mask]] = None
    dose: Optional[Dose] = None

    @property
    def reference_image(self) -> sitk.Image:
        """Returns a reference image used for spatial alignment across modalities.

        Priority order: CT → Dose → First available structure mask.

        Raises:
            ValueError: If no suitable reference image is available.
        """
        if self.ct and self.ct.image:
            return self.ct.image
        if self.dose and self.dose.image:
            return self.dose.image
        if self.structure_masks:
            first_mask = next(iter(self.structure_masks.values()))
            if first_mask.image:
                return first_mask.image
        raise ValueError("No reference image found in Case.")

    def get_mask_origin(self) -> Optional[NDArray[float]]:
        """Determines if all structure masks share a common origin.

        Returns:
            The common origin as a NumPy array if consistent; otherwise, None.
        """
        if not self.structure_masks:
            return None

        origins = {mask.image.GetOrigin() for mask in self.structure_masks.values() if mask.image}
        if len(origins) == 1:
            (unique_origin,) = origins
            return np.array(unique_origin)
        return None

    def resample(self, new_size: NDArray[int], new_spacing: NDArray[float], new_origin: NDArray[float]):
        """Resamples all images within the Case to a new spatial resolution and size.

        Args:
            new_size: Desired size of the resampled images (x, y, z).
            new_spacing: Desired spacing of the resampled images (x, y, z).
            new_origin: Desired origin of the resampled images (x, y, z).
        """
        size = tuple(new_size.astype(int).tolist())
        spacing = tuple(new_spacing.astype(float).tolist())
        origin = tuple(new_origin.astype(float).tolist())
        direction = self.reference_image.GetDirection()

        if self.ct and self.ct.image:
            self.ct.image = resample_img(self.ct.image, size, spacing, origin, direction)
        if self.dose and self.dose.image:
            self.dose.image = resample_img(self.dose.image, size, spacing, origin, direction)

        if self.structure_masks:
            for mask in self.structure_masks.values():
                if mask.image:
                    mask.image = resample_img(mask.image, size, spacing, origin, direction)

    def get_placeholder_image(self) -> sitk.Image:
        """Generates an empty placeholder image matching the reference image's spatial properties.

        Returns:
            An empty SimpleITK image with matching size, spacing, origin, and direction.
        """
        reference_image = self.reference_image
        placeholder = sitk.Image(reference_image.GetSize(), reference_image.GetPixelID())
        placeholder.CopyInformation(reference_image)
        return placeholder

    def __str__(self):
        return (
            f"Case(id={self.id}, type={self.type}, "
            f"ct={'yes' if self.ct else 'no'}, "
            f"dose={'yes' if self.dose else 'no'}, "
            f"masks={len(self.structure_masks) if self.structure_masks else 0})"
        )
