from enum import Enum
from typing import Optional

import cv2
import numpy as np
import SimpleITK as sitk
from case import Case
from case.attributes.contour import Contour
from case.attributes.mask import Mask
from numpy.typing import NDArray
from scipy.interpolate import interp1d

from prediction.sample_config import SampleConfig
from prediction.structure_names import StructureNames


class SupportedContoursTypes(Enum):
    closed_planar = "CLOSED_PLANAR"
    open_planar = "OPEN_NONPLANAR"


def create_masks(plan: Case, config: SampleConfig) -> None:
    """Create and standardize masks for each contoured structure."""
    structure_name_config = StructureNames.read()
    plan.type = config.type

    # Generate raw masks from contours.
    raw_structure_masks: dict[str, Mask] = {}
    for contour in plan.structure_contours.values():
        contour.snap_to_grid(plan.ct.image)
        mask_array = _create_boolean_mask(contour, reference_image=plan.ct.image)
        mask_image = _create_mask_image(mask_array, reference_image=plan.ct.image)
        raw_structure_masks[contour.name] = Mask(name=contour.name, image=mask_image)

    # Rename and merge masks with the same standardized name.
    plan.structure_masks = {}
    for mask in raw_structure_masks.values():
        mask_name = structure_name_config(mask.name)
        if mask_name in plan.structure_masks:
            plan.structure_masks[mask_name] += mask
        elif mask_name is not None:
            mask.name = mask_name
            mask.plan_type = plan.type
            plan.structure_masks[mask_name] = mask


def _create_mask_image(mask_array: NDArray[bool], reference_image: sitk.Image) -> sitk.Image:
    """Convert a boolean mask array to a SimpleITK image matching the reference geometry."""
    image = sitk.GetImageFromArray(mask_array.astype(np.uint8).T)
    image.SetSpacing(reference_image.GetSpacing())
    image.SetDirection(reference_image.GetDirection())
    image.SetOrigin(reference_image.GetOrigin())
    return image


def _create_boolean_mask(contour: Contour, reference_image: sitk.Image) -> Optional[NDArray[bool]]:
    """Create a boolean mask from a contour."""
    mask = np.zeros(reference_image.GetSize(), dtype=bool)
    for points, geometric_type in zip(contour.points, contour.geometric_types):
        indices = np.asarray([reference_image.TransformPhysicalPointToIndex(point) for point in points])
        _add_contour_slice_to_mask(indices, geometric_type, mask)
    return mask


def _add_contour_slice_to_mask(indices: NDArray[int], geometric_type: str, mask: NDArray[bool]) -> None:
    """Add a contour slice to the mask based on its geometric type."""
    row_nums, col_nums, slice_nums = indices.T
    if geometric_type == SupportedContoursTypes.closed_planar.value:
        num_slices = len(np.unique(slice_nums))
        if num_slices != 1:
            raise ValueError(f"Expected 1 slice, found {num_slices}.")
        mask_slice = _mask_slice_of_contour(row_nums, col_nums, mask.shape[:-1])
        mask[mask_slice, slice_nums[0]] = True
    elif geometric_type == SupportedContoursTypes.open_planar.value:
        all_rows, all_cols, all_slices = _interpolate_integer_points(row_nums, col_nums, slice_nums)
        mask[all_rows, all_cols, all_slices] = True
    else:
        raise ValueError(f"Unsupported contour type: {geometric_type}.")


def _mask_slice_of_contour(row_coords: NDArray[int], col_coords: NDArray[int], mask_shape: tuple[int, ...]) -> NDArray[bool]:
    """Generate a 2D mask from contour coordinates."""
    yx_contour = np.array([col_coords, row_coords]).T.reshape((1, -1, 2))
    mask = np.zeros(mask_shape, dtype=np.uint8)
    cv2.fillPoly(mask, yx_contour, 1)  # type: ignore
    return mask.astype(bool)


def _interpolate_integer_points(row_nums: NDArray[int], col_nums: NDArray[int], slice_nums: NDArray[int]) -> NDArray[int]:
    """Interpolate points for open planar contours."""
    interp_row = interp1d(slice_nums, row_nums)
    interp_col = interp1d(slice_nums, col_nums)
    all_z_vals = np.arange(slice_nums.min(), slice_nums.max() + 1)
    all_rows = np.round(interp_row(all_z_vals)).astype(int)
    all_cols = np.round(interp_col(all_z_vals)).astype(int)
    return np.array((all_rows, all_cols, all_z_vals))
