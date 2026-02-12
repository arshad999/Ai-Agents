from pydantic import BaseModel, Field

# Schema (how LLM understands tool input)
class MultiplyInput(BaseModel):
    a: float = Field(description="first number")
    b: float = Field(description="second number")

# Actual function
def multiply(a: float, b: float):
    return a * b
