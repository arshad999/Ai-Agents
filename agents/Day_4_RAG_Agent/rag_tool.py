from langchain.tools import tool
from rag_store import load_vectorstore
from graph_logger import GraphLogger

db = load_vectorstore()
logger = GraphLogger()

@tool
def search_docs(query: str) -> str:
    """Search company knowledge base"""
    logger.log("Search Docs", f"Query: {query}")
    docs = db.similarity_search(query, k=3)
    result = "\n".join([d.page_content for d in docs])
    logger.log("Search Result", result)
    return result
