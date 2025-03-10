import math
from itertools import chain

import numpy as np
from case import Case
from case.config import CaseConfig
from case.reader_writer.reader import read_case
from case.reader_writer.writer import write_case
from numpy.typing import NDArray
from tqdm import tqdm

from prediction.create_data_splits import create_data_splits
from prediction.mask_contours import create_masks
from prediction.sample import Sample
from prediction.sample_config import SampleConfig
from prediction.scaling import ScalingConfig
from prediction.structure_names import StructureNames
from prediction.validator import check_availability


def create_prediction_data(plan_type: str, base_type: str = "clinical") -> None:
    """
    Create prediction data using the specified plan type and base type.

    This function performs the following steps:
      1. Reads the prediction configuration.
      2. Checks available plans and creates train/validation/test splits.
      3. Scans training data to compute median voxel spacing and maximum bounding box size.
         It then determines a standardized bounding box size based on the smallest power of two.
      4. For each plan ID, reads the plan, creates standardized masks, resamples the image data
         to match the bounding box and voxel spacing, and writes the modified plan to disk.

    Args:
        plan_type: The prediction data plan type to create.
        base_type: The base data type to use for creating prediction data.
    """
    prediction_config = SampleConfig.read(plan_type)
    eligible_ids = check_availability(prediction_config, base_type)
    data_splits = create_data_splits(eligible_ids, val_test_ratio=0.1, test_ratio=0.2)
    data_splits.write(plan_type)

    voxel_spacing, max_bounding_box = _scan_image_dimensions(prediction_config, base_type, data_splits.train_ids)
    max_box_bound = np.max(max_bounding_box)
    min_valid_size = 2**6  # Based on a U-net that down-samples the input dimensions 6 times
    standard_image_size = min_valid_size * math.ceil(max_box_bound / min_valid_size)
    bounding_box = np.array((standard_image_size, standard_image_size, standard_image_size))
    case_config = CaseConfig.read(base_type)

    for plan_id in tqdm(data_splits.get_all_ids(), desc=f"Creating prediction data for {plan_type}"):
        case = read_case(plan_id, case_config=case_config)
        create_masks(case, prediction_config)
        origin = _get_bounding_box_origin(case, bounding_box, voxel_spacing)
        case.resample(bounding_box, voxel_spacing, origin)
        write_case(case)


def _scan_image_dimensions(prediction_config: SampleConfig, base_type: str, plan_ids: list[str]) -> tuple[NDArray[float], NDArray[int]]:
    """
    Compute the median voxel spacing and maximum bounding box size from training plans.

    For each plan in plan_ids, this function:
      - Reads the plan and extracts ROI points from structures specified in the prediction config.
      - Computes the minimum and maximum ROI coordinates.
      - Gathers voxel spacing from each plan's dose image.
      - Updates scaling configuration based on image features.

    Returns:
        A tuple (new_voxel_spacing, bounding_box_sizes) where:
          - new_voxel_spacing: Median voxel spacing across training plans.
          - bounding_box_sizes: Maximum dimensions (in voxel indices) of the bounding box.

    Args:
        prediction_config: Config specifying the input and output features.
        base_type: Base data type (e.g., "clinical") for reading plans.
        plan_ids: List of plan IDs to scan.
    """
    structure_names_mapper = StructureNames.read()
    structure_names = prediction_config.get_structure_names()
    scale_config = ScalingConfig.from_feature_names([*prediction_config.input_names, *prediction_config.output_names])
    min_roi_points = []
    max_roi_points = []
    voxel_spacing = []
    case_config = CaseConfig.read(base_type)

    for id_ in tqdm(plan_ids, desc="Scanning training data"):
        patient = read_case(id_, case_config)
        all_roi_points = []
        for structure, contour in patient.structure_contours.items():
            mapped_name = structure_names_mapper(contour.name)
            if mapped_name in structure_names:
                all_roi_points.extend(chain.from_iterable(contour.points))
        min_roi_points.append(np.min(all_roi_points, axis=0))
        max_roi_points.append(np.max(all_roi_points, axis=0))
        voxel_spacing.append(np.array(patient.dose.image.GetSpacing()).T)
        sample = Sample.from_patient(patient, prediction_config)
        for name, data in sample.get_image_features():
            scale_config.update(name, data)

    new_voxel_spacing = np.median(voxel_spacing, axis=0)
    min_roi_indices = np.floor(np.array(min_roi_points) / new_voxel_spacing.reshape(1, -1)).astype(int)
    max_roi_indices = np.ceil(np.array(max_roi_points) / new_voxel_spacing.reshape(1, -1)).astype(int)
    bounding_box_sizes = max_roi_indices - min_roi_indices
    scale_config.add_buffer(offset=0, scale=0)
    scale_config.write()

    return new_voxel_spacing, bounding_box_sizes.max(axis=0)


def _get_bounding_box_origin(plan: Case, bounding_box: NDArray[int], voxel_spacing: NDArray[float]) -> NDArray[float]:
    """
    Compute the origin of the bounding box for a plan.

    The function concatenates all ROI positions from the plan's structure masks,
    determines the minimum and maximum coordinates, and calculates the required padding
    to center the ROIs in the standardized bounding box. The resulting origin is the minimum
    ROI coordinate minus the computed padding.

    Args:
        plan: The case containing structure masks with ROI positions.
        bounding_box: Standardized bounding box dimensions (in voxels).
        voxel_spacing: Voxel spacing (in physical units).

    Returns:
        The origin of the bounding box in physical space.
    """
    all_roi_points = np.concatenate([mask.positions for mask in plan.structure_masks.values()])
    min_roi_points = all_roi_points.min(axis=0)
    max_roi_points = all_roi_points.max(axis=0)
    roi_box_size = max_roi_points - min_roi_points
    box_padding = (bounding_box * voxel_spacing - roi_box_size) / 2
    origin = min_roi_points - box_padding
    return origin
