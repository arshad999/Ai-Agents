from pydantic import BaseModel, Field
from typing import List

class Plan(BaseModel):
    steps: List[str] = Field(description="ordered steps to solve the task")
