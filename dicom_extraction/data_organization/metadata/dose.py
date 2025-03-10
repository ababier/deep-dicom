from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pydicom
from dicom_extraction.data_organization.metadata.base import DicomMetadata


@dataclass
class Dose(DicomMetadata):
    """
    Represents dose data extracted from a DICOM dataset.
    """

    ReferencedStructureSetSOPInstanceUID: Optional[str]
    FrameOfReferenceUID: str
    ReferencedPlanSOPInstanceUID: Optional[str] = None

    @classmethod
    def from_dicom(cls, ds: pydicom.Dataset, path: Path) -> Dose:
        """Create a Dose instance from a DICOM dataset."""
        return cls(
            path=path,
            SeriesInstanceUID=ds.SeriesInstanceUID,
            SOPInstanceUID=ds.SOPInstanceUID,
            StudyInstanceUID=ds.SeriesDescription,
            timestamp=ds.timestamp,
            ReferencedStructureSetSOPInstanceUID=ds.ReferencedStructureSetSequence[0].ReferencedSOPInstanceUID
            if "ReferencedStructureSetSequence" in ds.values()
            else None,
            ReferencedPlanSOPInstanceUID=ds.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID,
            FrameOfReferenceUID=ds.FrameOfReferenceUID,
        )
