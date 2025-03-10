import SimpleITK as sitk
from case import attributes

from dicom_extraction.data_organization.catalogue import Catalogue


def make_ct(dicom_catalogue: Catalogue, image_uid: str) -> attributes.CT:
    """Create a CT object from the DICOM catalogue entry."""
    ct_metadata = dicom_catalogue.images[image_uid]
    reader = sitk.ImageSeriesReader()
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    reader.SetOutputPixelType(sitk.sitkInt16)
    reader.SetFileNames([path.as_posix() for path in ct_metadata.files])
    image = reader.Execute()
    return attributes.CT(image=image)
