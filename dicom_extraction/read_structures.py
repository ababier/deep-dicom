import pydicom
from case.attributes.contour import Contour

from dicom_extraction.data_organization.catalogue import Catalogue


def make_contours(dicom_catalogue: Catalogue, structure_id: str) -> dict[str, Contour]:
    """
    Create a dictionary mapping ROI names to Contour objects for a given structure set.

    Args:
        dicom_catalogue: The catalogue containing DICOM metadata.
        structure_id: The identifier for the structure set.

    Returns:
        A dictionary of ROI contours. If no contour sequence is found, returns an empty dict.
    """
    structure_metadata = dicom_catalogue.structure_sets.get(structure_id)
    if structure_metadata is None:
        return {}

    structure_dicom = pydicom.read_file(structure_metadata.path)
    roi_contour_sequence = getattr(structure_dicom, "ROIContourSequence", None)
    if not roi_contour_sequence:
        return {}

    roi_contours: dict[str, Contour] = {}
    for roi_name, contour in zip(structure_metadata.roi_by_index.values(), roi_contour_sequence):
        contour_sequence = getattr(contour, "ContourSequence", None)
        if contour_sequence:
            roi_contours[roi_name] = Contour.from_sequence(name=roi_name, contour_sequence=contour_sequence)
    return roi_contours
