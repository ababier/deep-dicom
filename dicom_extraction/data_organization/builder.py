import logging
import os
from pathlib import Path

import pydicom
import SimpleITK as sitk
from dicom_extraction.data_organization.catalogue import Catalogue
from dicom_extraction.data_organization.metadata import Dose, Images, Plan, StructureSet

logger = logging.getLogger(__name__)


def build_dicom_catalogue(input_path: Path) -> Catalogue:
    """Build a DICOM catalogue from the given input path."""
    catalogue = Catalogue(patient_id=input_path.name)
    for root, _, files in os.walk(input_path):
        if any(file.lower().endswith(".dcm") for file in files):
            _set_dicom_meta_data(catalogue, Path(root))
    return catalogue


def _set_dicom_meta_data(catalogue: Catalogue, dicom_dir: Path) -> None:
    """Extract DICOM metadata from a directory and add it to the catalogue."""
    reader = sitk.ImageSeriesReader()
    reader.GlobalWarningDisplayOff()
    series_ids = reader.GetGDCMSeriesIDs(dicom_dir.as_posix())
    added_paths = set()

    for series_id in series_ids:
        series_paths = [Path(p) for p in reader.GetGDCMSeriesFileNames(dicom_dir.as_posix(), series_id)]
        _add_series_paths(catalogue, series_paths)
        added_paths.update(series_paths)

    # Process remaining DICOM files not part of a series.
    all_dicom_paths = {p for p in dicom_dir.iterdir() if p.suffix.lower() == ".dcm"}
    for path in all_dicom_paths - added_paths:
        _add_path(catalogue, path)


def _add_path(catalogue: Catalogue, path: Path) -> None:
    """Add metadata from a single DICOM file to the catalogue."""
    dicom = pydicom.read_file(path)
    modality = dicom.Modality.lower()
    if "struct" in modality:
        structure_set = StructureSet.from_dicom(dicom, path)
        catalogue.structure_sets[structure_set.SOPInstanceUID] = structure_set
    elif "read_plan" in modality:
        plan = Plan.from_dicom(dicom, path)
        catalogue.plans[plan.SOPInstanceUID] = plan
    else:
        logger.warning(f"{path} contains unsupported modality '{dicom.Modality}'. Supported: 'struct', 'read_plan'.")


def _add_series_paths(catalogue: Catalogue, series_paths: list[Path]) -> None:
    """Add metadata from a series of DICOM files to the catalogue."""
    first_path = series_paths[0]
    dicom = pydicom.read_file(first_path)
    modality = dicom.Modality.lower()

    if "dose" in modality:
        dose = Dose.from_dicom(dicom, first_path)
        catalogue.doses[dose.SOPInstanceUID] = dose
    elif "ct" in modality or "mr" in modality:
        images = Images.from_dicom(dicom, first_path.parent)
        images.add_files(series_paths)
        catalogue.images[images.SeriesInstanceUID] = images
    else:
        logger.warning(f"{first_path} contains unsupported modality '{dicom.Modality}'. Supported: 'dose', 'ct', 'mr'.")
