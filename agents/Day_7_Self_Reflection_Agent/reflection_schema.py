from pydantic import BaseModel, Field

class Critique(BaseModel):
    correct: bool = Field(description="whether the answer is correct")
    feedback: str = Field(description="what is wrong and how to fix")
