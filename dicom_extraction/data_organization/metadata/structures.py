from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pydicom
from dicom_extraction.data_organization.metadata.base import DicomMetadata


@dataclass
class StructureSet(DicomMetadata):
    roi_by_index: dict[int, str]
    FrameOfReferenceUID: str

    @classmethod
    def from_dicom(cls, ds: pydicom.Dataset, path: Path) -> StructureSet:
        """Create a StructureSet from a DICOM dataset."""
        structures = getattr(ds, "StructureSetROISequence", [])
        return cls(
            path=path,
            SeriesInstanceUID=cls._get_referred_series(ds),
            SOPInstanceUID=ds.SOPInstanceUID,
            StudyInstanceUID=ds.SeriesDescription,
            timestamp=ds.timestamp,
            roi_by_index={int(structure.ROINumber): structure.ROIName.lower() for structure in structures},
            FrameOfReferenceUID=ds.ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID,
        )

    @staticmethod
    def _get_referred_series(ds: pydicom.Dataset) -> str:
        """Extract the unique SeriesInstanceUID from the referenced series."""
        referred_series = [
            sequence.SeriesInstanceUID
            for reference_sequence in ds.ReferencedFrameOfReferenceSequence
            for study_sequence in getattr(reference_sequence, "RTReferencedStudySequence", [])
            for sequence in study_sequence.RTReferencedSeriesSequence
        ]
        if len(referred_series) != 1:
            raise ValueError(f"Expected 1 RTReferencedSeriesSequence, found {len(referred_series)}.")
        return referred_series[0]
