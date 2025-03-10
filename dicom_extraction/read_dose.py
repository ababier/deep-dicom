import SimpleITK as sitk
from case import attributes

from dicom_extraction.data_organization.catalogue import Catalogue


def make_dose(dicom_catalogue: Catalogue, dose_uid: str) -> attributes.Dose:
    """Create a Dose object from a DICOM catalogue entry using the dose scaling factor."""
    dose_path = dicom_catalogue.doses[dose_uid].path
    image = sitk.ReadImage(dose_path.as_posix())
    scaling_factor = float(image.GetMetaData("3004|000e"))  # Retrieve the dose scaling factor (DICOM tag 3004|000e) and apply it
    scaled_image = sitk.Multiply(sitk.Cast(image, sitk.sitkFloat64), scaling_factor)
    return attributes.Dose(image=scaled_image)
