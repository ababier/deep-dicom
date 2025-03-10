import json
from typing import Optional

import SimpleITK as sitk
from case import Case
from case.attributes import CT, Dose
from case.attributes.contour import Contour
from case.attributes.mask import Mask
from case.config import CaseConfig
from case.reader_writer import make_path


def read_case(id_: str, case_config: CaseConfig) -> Case:
    """
    Read a case from disk based on the provided case configuration.

    Args:
        id_: Identifier for the case.
        case_config: The configuration settings for the case.

    Returns:
        A Case object containing the CT, dose, structure contours, and structure masks as specified.
    """
    case_type = case_config.type
    structure_contour_names = case_config.get_structure_contour_names(id_)
    structure_mask_names = case_config.get_structure_mask_names(id_)

    return Case(
        id=id_,
        type=case_type,
        ct=read_ct(id_, case_type) if case_config.ct else None,
        dose=_read_dose(id_, case_type) if case_config.dose else None,
        structure_contours=_read_structure_contour(id_, structure_contour_names, case_type) if structure_contour_names else None,
        structure_masks=_read_structure_mask(id_, structure_mask_names, case_type) if structure_mask_names else None,
    )


def read_ct(id_: str, type_: str) -> Optional[CT]:
    path = make_path.ct(id_, type_)
    if path.exists():
        return CT(image=sitk.ReadImage(path.as_posix()))
    return None


def _read_dose(id_: str, type_: str) -> Optional[Dose]:
    path = make_path.dose(id_, type_)
    if path.exists():
        return Dose(image=sitk.ReadImage(path.as_posix()))
    return None


def _read_structure_contour(id_: str, names: list[str], case_type: str) -> Optional[dict[str, Contour]]:
    contours: dict[str, Contour] = {}
    for name in names:
        path = make_path.structure_contour(name, id_, case_type)
        with open(path) as file:
            data = json.loads(file.read())
        contours[name] = Contour(**data)
    return contours or None


def _read_structure_mask(id_: str, names: list[str], case_type: str) -> dict[str, Mask]:
    structure_masks = {name: Mask(image=sitk.ReadImage(make_path.structure_mask(name, id_, case_type).as_posix()), name=name) for name in names}
    return structure_masks or None
