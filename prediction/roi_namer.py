from typing import Optional

import Levenshtein
from case.reader_writer import make_path
from thefuzz import process

from prediction.structure_names import StructureNames


def get_all_contour_names(plan_type: str) -> set[str]:
    """Return all contour names for the given plan type."""
    structure_names = set()
    for path in make_path.case_parents(plan_type).iterdir():
        id_ = path.name
        structure_names.update(make_path.get_all_contour_names(id_=id_, type_=plan_type))
    return structure_names


def consolidate_names(names: set[str], name_map: Optional[StructureNames] = None, threshold: int = 95) -> StructureNames:
    """Group similar names using edit distance."""
    if name_map:
        ungrouped_labels = _build_on_existing_group(names, name_map, threshold)
        if ungrouped_labels:
            name_map.mapping |= _create_group_from_scratch(ungrouped_labels, threshold).mapping
    else:
        name_map = _create_group_from_scratch(names, threshold)
    return name_map


def _build_on_existing_group(names: set[str], name_map: StructureNames, threshold: int) -> set[str]:
    """Add similar names to existing groups; return names not grouped."""
    known_labels = set(name_map.mapping)
    ungrouped_labels = set()
    edit_distance_threshold = 100 - threshold
    for name in names:
        if name not in name_map.mapping:
            distances = {fixed: Levenshtein.distance(name.lower(), fixed) for fixed in known_labels}
            label = min(distances, key=distances.get)
            if distances[label] < edit_distance_threshold:
                name_map.mapping[label].add(name)
            else:
                ungrouped_labels.add(name)
    return ungrouped_labels


def _create_group_from_scratch(names: set[str], threshold: int) -> StructureNames:
    """Create new groups from names using fuzzy matching."""
    groups: dict[str, set[str]] = {}
    while names:
        group_name = names.pop()
        matches = process.extractBests(group_name, names, score_cutoff=threshold, limit=None)
        matched_names = {group_name, *(name for name, _ in matches)}
        names.difference_update(matched_names)
        groups[group_name] = matched_names
    return StructureNames(mapping=groups)
