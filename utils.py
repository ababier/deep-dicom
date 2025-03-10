import numpy as np
import SimpleITK as sitk
from numpy.typing import NDArray

# Mapping SimpleITK Pixel Types to NumPy Data Types
SITK_TO_NUMPY_TYPE = {
    sitk.sitkUInt8: np.uint8,
    sitk.sitkInt8: np.int8,
    sitk.sitkUInt16: np.uint16,
    sitk.sitkInt16: np.int16,
    sitk.sitkUInt32: np.uint32,
    sitk.sitkInt32: np.int32,
    sitk.sitkUInt64: np.uint64,
    sitk.sitkInt64: np.int64,
    sitk.sitkFloat32: np.float32,
    sitk.sitkFloat64: np.float64,
}


def sitk_image_to_array(image: sitk.Image) -> NDArray:
    pixel_type = SITK_TO_NUMPY_TYPE[image.GetPixelIDValue()]
    return sitk.GetArrayFromImage(image).astype(pixel_type)
