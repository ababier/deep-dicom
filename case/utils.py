from __future__ import annotations

from typing import Optional

import numpy as np
import SimpleITK as sitk
from numpy._typing import NDArray


def resample_img(
    image: sitk.Image,
    size: tuple[int, ...],
    spacing: tuple[float, ...],
    origin: tuple[float, ...],
    direction: tuple[float, ...],
) -> sitk.Image:
    """
    Resample a SimpleITK image to new geometry.

    Args:
        image: Input image.
        size: Desired output size.
        spacing: Desired voxel spacing.
        origin: Desired origin.
        direction: Desired direction cosines.

    Returns:
        Resampled image. Uses nearest neighbor interpolation for 8-bit images,
        otherwise B-Spline interpolation.
    """
    dim = image.GetDimension()

    # Adjust the origin to account for flips.
    # For each axis, if the corresponding diagonal element in the desired
    # direction is negative, shift the origin by (size - 1) * spacing along that axis.
    adjusted_origin = list(origin)
    for i in range(dim):
        # In a flattened direction matrix, the (i,i) element is at index i*dim + i.
        if direction[i * dim + i] < 0:
            adjusted_origin[i] = origin[i] + (size[i] - 1) * spacing[i]

    resample = sitk.ResampleImageFilter()
    resample.SetSize(size)
    resample.SetOutputSpacing(spacing)
    resample.SetOutputOrigin(adjusted_origin)
    resample.SetOutputDirection(direction)
    resample.SetTransform(sitk.Transform())

    if image.GetPixelIDTypeAsString() == "8-bit unsigned integer":
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)
    new_image = resample.Execute(image)
    return new_image


def indices_to_physical_points(base_image: sitk.Image, indices: Optional[NDArray[int]] = None) -> NDArray[float]:
    """
    Convert voxel indices to physical coordinates using a reference image.

    Args:
        base_image: Reference image.
        indices: Array of voxel indices (shape (N, D)). If None, all non-zero indices are used.

    Returns:
        Array of physical coordinates (shape (N, D)).
    """
    if indices is None:
        indices = _get_non_zero_indices(base_image).T

    origin = np.array(base_image.GetOrigin())
    spacing = np.array(base_image.GetSpacing())
    direction = np.array(base_image.GetDirection()).reshape(base_image.GetDimension(), base_image.GetDimension())
    return origin + (indices * spacing) @ direction


def _get_non_zero_indices(image: sitk.Image) -> NDArray[int]:
    """
    Get indices of non-zero voxels in (x, y, z) order.

    Args:
        image: A SimpleITK image.

    Returns:
        Array of shape (3, N) with non-zero voxel indices.
    """
    image_array = sitk.GetArrayViewFromImage(image)
    nonzero_indices = np.argwhere(image_array)  # (z, y, x) order.
    if nonzero_indices.size == 0:
        return np.empty((3, 0), dtype=int)
    z, y, x = nonzero_indices.T
    return np.vstack((x, y, z))
