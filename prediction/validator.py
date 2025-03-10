from case.reader_writer import make_path

from prediction.sample_config import SampleConfig
from prediction.structure_names import StructureNames


def check_availability(prediction_config: SampleConfig, base_data: str) -> list[str]:
    """
    Return case IDs that have the required data for the given config.
    """
    all_case_ids = [path.name for path in make_path.case_parents(base_data).iterdir()]
    return [id_ for id_ in all_case_ids if _check_case(prediction_config, base_data, id_)]


def _check_case(config: SampleConfig, base_data: str, id_: str) -> bool:
    """
    Check if a case with the given id has all mandatory data.

    Returns False if any required modality or structure is missing.
    """
    structure_name_config = StructureNames.read()

    if config.is_mandatory("ct") and not make_path.ct(id_, base_data).exists():
        return False

    if config.is_mandatory("dose") and not make_path.dose(id_, base_data).exists():
        return False

    # Check mandatory structure contours
    for structure in config.get_structure_names(only_mandatory=True):
        possible_names = structure_name_config.mapping.get(structure, set())
        if not any(make_path.structure_contour(name, id_, base_data).exists() for name in possible_names):
            return False

    return True
