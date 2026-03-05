from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

# ─────────────────────────────────────────────────────────────────
# Request Models
# ─────────────────────────────────────────────────────────────────

class FileIngestMetadata(BaseModel):
    """Metadata to attach to an uploaded policy PDF"""
    policy_name: str
    insurer: str
    policy_type: str = "health"
    policy_number: Optional[str] = ""
    sum_insured: Optional[str] = ""
    premium: Optional[str] = ""
    holder_name: Optional[str] = ""
    tags_string: Optional[str] = ""  # Comma separated list of tags

class QueryRequest(BaseModel):
    """Schema for general chat query"""
    query: str = Field(..., description="The user's question")
    policy_name: Optional[str] = Field(None, description="Optional specific policy to search")
    language: str = Field("english", description="Response language e.g. 'hindi', 'tamil'")
    include_images: bool = True
    max_tokens: int = 1024

class ComparePoliciesRequest(BaseModel):
    """Schema for comparing two policies side-by-side"""
    policy_name_a: str
    policy_name_b: str
    query: str
    language: str = "english"

class ChecklistRequest(BaseModel):
    """Schema for generating a claim checklist"""
    query: str
    policy_name: Optional[str] = None
    language: str = "english"

class ExclusionRequest(BaseModel):
    """Schema for retrieving all exclusions"""
    query: str = "What is not covered in my policy?"
    policy_name: Optional[str] = None
    language: str = "english"

class IsExcludedRequest(BaseModel):
    """Schema for a quick yes/no exclusion lookup"""
    item: str
    policy_name: Optional[str] = None

# ... you can define response models similarly later.
