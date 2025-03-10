from pathlib import Path

from case import Case
from case.reader_writer.writer import write_case
from tqdm import tqdm

from dicom_extraction.data_organization.builder import build_dicom_catalogue
from dicom_extraction.read_ct import make_ct
from dicom_extraction.read_dose import make_dose
from dicom_extraction.read_structures import make_contours


def make_dicom_interfaces(patients_parent_path: Path, plan_type: str = "clinical") -> None:
    """
    Build and write case interfaces for all patients found in the given directory.

    Args:
        patients_parent_path: Parent directory containing individual patient DICOM directories.
        plan_type: The type/category for the case (default is "clinical").
    """
    # Get all patient directories, skipping hidden ones.
    patient_paths = [p for p in patients_parent_path.iterdir() if not p.name.startswith(".")]

    for patient_path in tqdm(patient_paths, desc="Processing DICOM cases", unit=" case"):
        dicom_catalogue = build_dicom_catalogue(patient_path)

        # Iterate through structure sets in the catalogue.
        for structures_uid, structure_set in dicom_catalogue.structure_sets.items():
            image_uid = structure_set.SeriesInstanceUID
            dose_id = dicom_catalogue.get_dose_id(structures_uid)
            case_interface = Case(
                id=f"{dicom_catalogue.patient_id}_{image_uid}",
                type=plan_type,
                ct=make_ct(dicom_catalogue, image_uid),
                structure_contours=make_contours(dicom_catalogue, structures_uid),
                dose=make_dose(dicom_catalogue, dose_id) if dose_id else None,
            )
            write_case(case_interface)
