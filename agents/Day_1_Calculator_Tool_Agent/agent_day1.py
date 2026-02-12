import json
from openai import OpenAI
from dotenv import load_dotenv
from tools import multiply, MultiplyInput
from logger import AgentLogger

load_dotenv()
client = OpenAI()
logger = AgentLogger()

# Tool definition
tools = [
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiply two numbers accurately. Use this for ANY math calculation.",
            "parameters": MultiplyInput.model_json_schema(),
        },
    }
]

user_question = "what is 3 * 5?"

messages = [
    {"role": "system", "content": "You are a reasoning agent. Think carefully. For math ALWAYS use tools."},
    {"role": "user", "content": user_question}
]

logger.log("USER QUESTION", user_question)

# First LLM decision
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools,
    tool_choice="auto",
)

msg = response.choices[0].message

# Log raw decision
logger.log("MODEL RAW RESPONSE", str(msg))

# If tool called
if msg.tool_calls:
    tool_call = msg.tool_calls[0]
    tool_name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)

    logger.log("TOOL SELECTED", tool_name)
    logger.log("TOOL ARGUMENTS", args)

    # Execute tool
    result = multiply(**args)
    logger.log("TOOL RESULT", result)

    # Send result back to model
    messages.append(msg)
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": str(result)
    })

    final = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )

    final_answer = final.choices[0].message.content
    logger.log("FINAL ANSWER", final_answer)

    print("\nAgent:", final_answer)

else:
    logger.log("NO TOOL USED", msg.content)
    print(msg.content)
