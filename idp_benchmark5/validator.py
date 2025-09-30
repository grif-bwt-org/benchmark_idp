import logging
from typing import Union, Optional, Dict, Any, List
from datetime import date
from pydantic import BaseModel, Field, ValidationError, field_validator

logger = logging.getLogger(__name__)

UNIT_MAP = {
    "pcs": "pc",
    "piece": "pc",
    "pieces": "pc",
    "set": "set",
    "sets": "set",
    "kg": "kg",
    "kgs": "kg",
    "m": "m",
    "meter": "m",
    "meters": "m",
    "box": "box",
}

class LineItem(BaseModel):
    line_item_number: int
    item_description: str
    category: Optional[str] = Field(None, description="A broad category for the item, e.g., 'fasteners', 'electrical components', 'safety equipment'.")
    item_details: Optional[str] = Field(None, description="Detailed specifications found directly under the item description.")
    special_instructions: Optional[str] = Field(None, description="Specific instructions for this item, often from a separate notes section.")
    part_number: Optional[str] = None
    quantity: Optional[float] = None
    unit: Optional[str] = None
    unit_price: Optional[float] = None
    total_price: Optional[float] = None
    key_value_attributes: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("unit", mode='before')
    @classmethod
    def normalize_unit(cls, v: Any) -> str:
        if isinstance(v, str):
            v_lower = v.lower().strip()
            return UNIT_MAP.get(v_lower, v_lower)
        return str(v)

class DocumentMetadata(BaseModel):
    document_type: str
    document_id: Optional[str] = None
    vessel_name: Optional[str] = None
    buyer_company: Optional[str] = None
    vendor_company: Optional[str] = None
    issue_date: Optional[date] = None
    currency: Optional[str] = None
    key_value_attributes: Dict[str, Any] = Field(default_factory=dict)

class ValidatedInvoice(BaseModel):
    document_metadata: DocumentMetadata
    line_items: List[LineItem]

def validate_invoice_data(data: Dict[str, Any]) -> Optional[ValidatedInvoice]:
    try:
        validated_model = ValidatedInvoice.model_validate(data)
        logger.debug("Pydantic validation was successful")
        return validated_model
    except ValidationError as e:
        logger.error(f"Error was occured in Pydantic: {e.errors()}")
        return None