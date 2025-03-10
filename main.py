import logging

import click

from dicom_extraction import make_dicom_interfaces
from globals import DATA_DIR
from logger import setup_logging
from models.trainer import Trainer
from prediction.process_sample_data import create_prediction_data
from prediction.roi_namer import consolidate_names, get_all_contour_names
from prediction.sample_config import SampleConfig
from prediction.structure_names import StructureNames

logger = logging.getLogger(__name__)

# Set up logging
setup_logging(log_file="deep-dicom.log", level=logging.DEBUG)

# Process DICOM data into images
dicom_plan_type = "clinical"  # Loaded attributes defined in configs/clinical.yaml
make_dicom_interfaces(DATA_DIR / "pancreas-sample", dicom_plan_type)

# Collect all contour names and make name mapping config
structure_names = get_all_contour_names(dicom_plan_type)
initial_groups = StructureNames.read() if StructureNames.get_path().exists() else None
name_mapping = consolidate_names(structure_names, initial_groups)
indented_mapping = "\n".join(f"\t{line}" for line in str(name_mapping).splitlines())
logger.warning(f"Standardized plans will use the following structure name mapping (new name: original names):\n{indented_mapping}")
name_mapping.write()
click.confirm(f"Check the structure mapping file {StructureNames.get_path()}. Press <Enter> to proceed")

# Process standardized data into format for dose prediction
prediction_name = "dose_prediction"
create_prediction_data(prediction_name)
prediction_config = SampleConfig.read("dose_prediction")
dose_trainer = Trainer(prediction_name)
dose_trainer.train_epochs()
