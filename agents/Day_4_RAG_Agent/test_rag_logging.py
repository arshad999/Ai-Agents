from rag_tool import search_docs
import os

print("Running search_docs test...")
try:
    result = search_docs.invoke("company policy")
    print("Search result length:", len(result))
    print("Test completed successfully.")
except Exception as e:
    print(f"Error: {e}")
