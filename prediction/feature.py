from typing import Optional

from pydantic import BaseModel


class Feature(BaseModel):
    """Represents a feature with optional sub-features."""

    name: str
    sub_features: Optional[list[str]] = None

    def get_complete_name(self, sub_feature: Optional[str] = None) -> str:
        """Return the complete name, appending a sub-feature if provided."""
        return f"{self.name}_{sub_feature}" if sub_feature else self.name
