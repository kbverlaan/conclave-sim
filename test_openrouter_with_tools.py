import os
from openai import OpenAI
from dotenv import load_dotenv
import json
import time

# Load environment variables from .env file
load_dotenv()

def test_openrouter_with_tools():
    # Get API key
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OpenRouter API key not found in environment variables")
        return

    # Initialize client
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )

    # Define the same tools we use in the main simulation
    tools = [
        {
            "type": "function",
            "function": {
                "name": "vote",
                "description": "Cast your vote for one agent (cannot be yourself)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reasoning": {
                            "type": "string",
                            "description": "Explain why you chose this agent"
                        },
                        "agent_id": {
                            "type": "integer",
                            "description": "The ID of the agent you're voting for"
                        }
                    },
                    "required": ["agent_id", "reasoning"]
                }
            }
        }
    ]

    # Test message with a similar structure to our simulation
    test_prompt = """You are Cardinal John. Here is some information about yourself: You are a conservative cardinal from the United States.
You are currently participating in the conclave to decide the next pope. The candidate that secures a 2/3 supermajority of votes wins.
The candidates are:
Cardinal 0 - John Smith
Cardinal 1 - Michael Brown
Cardinal 2 - Robert Davis

Use your vote tool to cast your vote for one of the candidates."""

    messages = [
        {"role": "user", "content": test_prompt}
    ]

    max_retries = 3
    retry_delay = 1

    for attempt in range(max_retries):
        try:
            print(f"\nAttempt {attempt + 1} of {max_retries}")
            print("Sending test request to OpenRouter...")
            
            response = client.chat.completions.create(
                model="anthropic/claude-3.7-sonnet",
                messages=messages,
                tools=tools,
                tool_choice={"type": "any"},
                max_tokens=1000,
                temperature=0.5
            )
            
            print("\nSuccess! Response received:")
            print(f"Model: {response.model}")
            if hasattr(response.choices[0].message, 'tool_calls'):
                tool_calls = response.choices[0].message.tool_calls
                print(f"Tool calls: {json.dumps(tool_calls, indent=2)}")
            else:
                print(f"Message: {response.choices[0].message.content}")
            break
            
        except Exception as e:
            print(f"\nError occurred on attempt {attempt + 1}:")
            print(f"Type: {type(e).__name__}")
            print(f"Message: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print("Max retries reached, giving up.")

if __name__ == "__main__":
    test_openrouter_with_tools()
