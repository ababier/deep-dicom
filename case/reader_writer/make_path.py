from pathlib import Path

from globals import PROCESSED_DATA_DIR


def case_parents(type_: str) -> Path:
    """Return the parent directory for plans of a given type."""
    return PROCESSED_DATA_DIR / type_


def _path_template(name: str, extension: str, id_: str, type_: str, parent: Path = Path("")) -> Path:
    """
    Build a file path based on provided parameters.

    If `name` is empty, returns the directory path.
    """
    filename = f"{name}{extension}" if name else ""
    return PROCESSED_DATA_DIR / type_ / id_ / parent / filename


def ct(id_: str, type_: str) -> Path:
    """Return the path to the CT image (in NIfTI format) for the given id and type."""
    return _path_template(name="ct", extension=".nii.gz", id_=id_, type_=type_)


def dose(id_: str, type_: str) -> Path:
    """Return the path to the dose image (in NIfTI format) for the given id and type."""
    return _path_template(name="dose", extension=".nii.gz", id_=id_, type_=type_)


def structure_contour(name: str, id_: str, type_: str) -> Path:
    """Return the path to a structure contour JSON file for the given id, type, and structure name."""
    return _path_template(name=name, extension=".json", id_=id_, type_=type_, parent=Path("structure_contours"))


def get_all_contour_names(id_: str, type_: str) -> set[str]:
    """
    Return a set of all contour names for the given id and type.

    It does so by listing all files in the structure contours directory and returning their stem names.
    """
    structure_contours_path = structure_contour("", id_, type_)
    return {path.stem for path in structure_contours_path.iterdir()}


def structure_mask(name: str, id_: str, type_: str) -> Path:
    """Return the path to a structure mask (in NIfTI format) for the given id, type, and structure name."""
    return _path_template(name=name, extension=".nii.gz", id_=id_, type_=type_, parent=Path("structure_masks"))


def get_all_mask_names(id_: str, type_: str) -> set[str]:
    """
    Return a set of all mask names for the given id and type.

    It does so by listing all files in the structure masks directory and stripping the extension.
    """
    structure_masks_path = structure_mask("", id_, type_)
    return {path.name.replace(".nii.gz", "") for path in structure_masks_path.iterdir()}
