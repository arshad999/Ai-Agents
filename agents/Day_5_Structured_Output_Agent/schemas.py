from pydantic import BaseModel, Field
from typing import Optional

class UserIntent(BaseModel):
    intent: str = Field(description="type of user request")
    date: Optional[str] = Field(description="date if present")
    item: Optional[str] = Field(description="item name if shopping")
    quantity: Optional[int] = Field(description="quantity of item")
