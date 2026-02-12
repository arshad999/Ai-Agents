from langchain.tools import tool
from graph_logger import GraphLogger

logger = GraphLogger()

@tool
def multiply(a: float, b: float) -> float:
    """Multiply numbers when math calculation is required"""
    logger.log("TOOL CALLED", "multiply")
    logger.log("TOOL INPUT", {"a": a, "b": b})
    result = a * b
    logger.log("TOOL OUTPUT", result)
    return result


@tool
def string_length(text: str) -> int:
    """Count number of characters in text"""
    logger.log("TOOL CALLED", "string_length")
    logger.log("TOOL INPUT", {"text": text})
    result = len(text)
    logger.log("TOOL OUTPUT", result)
    return result
