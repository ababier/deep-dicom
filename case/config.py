from __future__ import annotations

from typing import Optional

import yaml
from globals import CONFIG_DIR
from pydantic import BaseModel

from case.reader_writer.make_path import get_all_contour_names, get_all_mask_names


class CaseConfig(BaseModel, extra="forbid"):
    """
    Configuration for a case.

    Attributes:
        type: Case category (e.g., 'clinical', 'dose prediction').
        ct: Whether to load CT images.
        dose: Whether to load dose images.
        structure_contours: List of contour names to process ("ALL" means all available).
        structure_masks: List of mask names to process ("ALL" means all available).
    """

    type: str
    ct: bool = False
    dose: bool = False
    structure_contours: Optional[list[str]] = None
    structure_masks: Optional[list[str]] = None

    def get_structure_contour_names(self, case_id: str) -> Optional[list[str]]:
        """Return contour names based on config; if 'ALL' is specified, return all available."""
        if self.structure_contours is None:
            return None

        available_contours = get_all_contour_names(case_id, self.type)
        if "ALL" in self.structure_contours:
            return list(available_contours)
        return list(available_contours.intersection(self.structure_contours))

    def get_structure_mask_names(self, case_id: str) -> Optional[list[str]]:
        """Return mask names based on config; if 'ALL' is specified, return all available."""
        if self.structure_masks is None:
            return None

        available_masks = get_all_mask_names(case_id, self.type)
        if "ALL" in self.structure_masks:
            return list(available_masks)
        return list(available_masks.intersection(self.structure_masks))

    @classmethod
    def read(cls, config_name: str) -> CaseConfig:
        """Load configuration from a YAML file."""
        config_path = CONFIG_DIR / f"{config_name}.yaml"
        with open(config_path, "r") as stream:
            config_data = yaml.safe_load(stream)
        return cls(**config_data)
