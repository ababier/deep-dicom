from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk
from pydicom import Sequence


@dataclass(kw_only=True)
class Contour:
    """
    Represents a medical image contour extracted from DICOM data.

    Attributes:
        name: Identifier of the contour.
        points: A list of contours, each a list of 3D points ([x, y, z]).
        geometric_types: Geometric types for each contour (e.g., "CLOSED_PLANAR").
    """

    name: str
    points: list[list[list[float]]]
    geometric_types: list[str]

    @classmethod
    def from_sequence(cls, name: str, contour_sequence: Sequence) -> Contour:
        """
        Build a Contour from a DICOM contour sequence.
        """
        contour_points = []
        geometric_types = []
        for dicom_contour in contour_sequence:
            pts = np.asarray(dicom_contour.ContourData).reshape((-1, 3))
            contour_points.append(pts.tolist())
            geometric_types.append(dicom_contour.ContourGeometricType)
        return cls(name=name, points=contour_points, geometric_types=geometric_types)

    def snap_to_grid(self, reference_image: sitk.Image) -> None:
        """
        Snap contour points to the grid of the provided reference image.
        """
        snapped_points = []
        for contour in self.points:
            indices = [reference_image.TransformPhysicalPointToIndex(pt) for pt in contour]
            snapped_contour = [reference_image.TransformIndexToPhysicalPoint(idx) for idx in indices]
            snapped_points.append(snapped_contour)
        self.points = snapped_points
