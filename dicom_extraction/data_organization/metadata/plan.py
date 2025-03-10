from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pydicom
from dicom_extraction.data_organization.metadata.base import DicomMetadata


@dataclass
class Plan(DicomMetadata):
    ReferencedStructureSetSOPInstanceUID: str

    @classmethod
    def from_dicom(cls, ds: pydicom.Dataset, path: Path) -> Plan:
        """Create a Plan instance from a DICOM dataset."""
        return cls(
            path=path,
            SeriesInstanceUID=ds.SeriesInstanceUID,
            SOPInstanceUID=ds.SOPInstanceUID,
            StudyInstanceUID=getattr(ds, "SeriesDescription", None),
            timestamp=ds.timestamp,
            ReferencedStructureSetSOPInstanceUID=ds.ReferencedStructureSetSequence[0].ReferencedSOPInstanceUID,
        )
