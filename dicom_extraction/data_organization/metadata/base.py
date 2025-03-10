from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pydicom


@dataclass
class DicomMetadata:
    """
    Base class for DICOM metadata.
    """

    path: Path
    SeriesInstanceUID: str
    SOPInstanceUID: str
    StudyInstanceUID: str
    timestamp: float

    @classmethod
    def from_dicom(cls, ds: pydicom.Dataset, path: Path) -> DicomMetadata:
        """
        Create an instance from a DICOM dataset.

        This method should be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses must implement from_dicom")
