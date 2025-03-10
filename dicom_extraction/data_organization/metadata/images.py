from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pydicom
from dicom_extraction.data_organization.metadata.base import DicomMetadata


@dataclass
class Images(DicomMetadata):
    Description: Optional[str]
    slice_thickness: float
    pixel_spacing_x: float
    pixel_spacing_y: float
    files: list[Path]
    PatientID: str

    @classmethod
    def from_dicom(cls, ds: pydicom.Dataset, path: Path) -> Images:
        """Create a Images instance from a DICOM dataset."""
        return cls(
            path=path,
            SeriesInstanceUID=ds.SeriesInstanceUID,
            SOPInstanceUID=ds.SOPInstanceUID,
            StudyInstanceUID=ds.StudyInstanceUID,
            timestamp=ds.timestamp,
            Description=getattr(ds, "StudyDescription", None),
            slice_thickness=float(ds.SliceThickness),
            pixel_spacing_x=float(ds.PixelSpacing[0]),
            pixel_spacing_y=float(ds.PixelSpacing[1]),
            files=[],
            PatientID=ds.PatientID,
        )

    def add_files(self, file_paths: Optional[list[Path]]) -> None:
        """Add file paths to the Images instance."""
        if file_paths:
            self.files.extend(file_paths)
