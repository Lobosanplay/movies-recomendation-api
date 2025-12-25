from pydantic import BaseModel
from typing import List, Optional

class TagsRequest(BaseModel):
    """Modelo para request con tags."""
    tags: List[str]
    limit: Optional[int] = 5