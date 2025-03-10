import dataclasses
import json
import os
from pathlib import Path

import SimpleITK as sitk
from case import Case
from case.attributes import CT, Dose
from case.attributes.contour import Contour
from case.attributes.mask import Mask
from case.reader_writer import make_path


def write_case(patient: Case) -> None:
    if patient.ct:
        _write_ct(patient.ct, patient.id, patient.type)
    if patient.dose:
        _write_dose(patient.dose, patient.id, patient.type)
    if patient.structure_contours:
        for contour in patient.structure_contours.values():
            _write_structure_contour(contour, patient.id, patient.type)
    if patient.structure_masks:
        for mask in patient.structure_masks.values():
            _write_structure_masks(mask, patient.id, patient.type)


def _write_ct(ct: CT, plan_id: str, plan_type: str) -> None:
    path = make_path.ct(plan_id, plan_type)
    os.makedirs(path.parent, exist_ok=True)
    _write_image(ct.image, path)


def _write_dose(dose: Dose, plan_id: str, plan_type: str) -> None:
    path = make_path.dose(plan_id, plan_type)
    os.makedirs(path.parent, exist_ok=True)
    _write_image(dose.image, path)


def _write_structure_masks(structure: Mask, plan_id: str, plan_type: str) -> None:
    path = make_path.structure_mask(structure.name, plan_id, plan_type)
    os.makedirs(path.parent, exist_ok=True)
    _write_image(structure.image, path)


def _write_structure_contour(contour: Contour, plan_id: str, plan_type: str) -> None:
    path = make_path.structure_contour(contour.name, plan_id, plan_type)
    os.makedirs(path.parent, exist_ok=True)
    with open(path.as_posix(), "w") as f:
        json.dump(dataclasses.asdict(contour), f)


def _write_image(image: sitk.Image, path: Path):
    os.makedirs(path.parent, exist_ok=True)
    path_as_str = path.as_posix()
    sitk.WriteImage(image, path_as_str)
