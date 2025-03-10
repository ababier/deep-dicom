import logging
from dataclasses import dataclass, field
from typing import Optional

from dicom_extraction.data_organization.metadata import Dose, Images, Plan, StructureSet

logger = logging.getLogger(__name__)


@dataclass
class Catalogue:
    patient_id: str
    images: dict[str, Images] = field(default_factory=dict)
    structure_sets: dict[str, StructureSet] = field(default_factory=dict)
    plans: dict[str, Plan] = field(default_factory=dict)
    doses: dict[str, Dose] = field(default_factory=dict)

    def get_dose_id(self, structures_uid: str) -> Optional[str]:
        """
        Return the SOPInstanceUID of the most recent dose associated with the given structure UID.

        If a plan is available, doses matching its ReferencedPlanSOPInstanceUID are used.
        Otherwise, doses are filtered based on matching FrameOfReferenceUID.
        """
        if not self.doses:
            return None

        plan_id = self.get_latest_plan_id(structures_uid)
        if plan_id:
            matching_doses = [dose for dose in self.doses.values() if dose.ReferencedPlanSOPInstanceUID == plan_id]
        else:
            # Fallback to matching frame of reference if no plan id is available
            structures = self.structure_sets[structures_uid]
            matching_doses = [dose for dose in self.doses.values() if dose.FrameOfReferenceUID == structures.FrameOfReferenceUID]

        if len(matching_doses) > 1:
            logger.warning(
                f"Multiple doses were found for structure UID {structures_uid} " f"of patient {self.patient_id}; returning the most recent dose."
            )

        last_updated_dose = max(matching_doses, key=lambda d: d.timestamp, default=None)
        return last_updated_dose.SOPInstanceUID if last_updated_dose else None

    def get_latest_plan_id(self, structure_uid: str) -> Optional[str]:
        """
        Return the SOPInstanceUID of the most recent plan associated with the given structure UID.
        """
        rt_plans = (rt_plan for rt_plan in self.plans.values() if rt_plan.ReferencedStructureSetSOPInstanceUID == structure_uid)
        latest_plan = max(rt_plans, key=lambda p: p.timestamp, default=None)
        return latest_plan.SOPInstanceUID if latest_plan else None
