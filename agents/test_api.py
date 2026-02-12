from dotenv import load_dotenv
import os
from langchain_openai import OpenAI

load_dotenv()

llm = OpenAI(
    model="gpt-4o-mini",
    temperature=0.7
)

response = llm.invoke("Explain gradient descent in simple words")
print(response)
