import os
from openai import OpenAI
from dotenv import load_dotenv
import json
import time

# Load environment variables from .env file
load_dotenv()

def test_tool_formats():
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

    # Function format based on OpenAI's format
    vote_tool = {
        "type": "function",
        "function": {
            "name": "cast_vote",
            "description": "Cast a vote for a candidate",
            "parameters": {
                "type": "object",
                "properties": {
                    "candidate": {
                        "type": "integer",
                        "description": "The ID of the candidate to vote for (0-2)"
                    }
                },
                "required": ["candidate"]
            }
        }
    }

    # Test cases with different tool configurations
    test_cases = [
        {
            "name": "1. Basic prompt without tools",
            "messages": [{"role": "user", "content": "Say hello!"}],
            "use_tools": False
        },
        {
            "name": "2. Simple function call",
            "messages": [{"role": "user", "content": "Please vote for candidate 1"}],
            "tools": [vote_tool],
            "tool_choice": None
        },
        {
            "name": "3. Forced function call",
            "messages": [{"role": "user", "content": "Please vote for candidate 1"}],
            "tools": [vote_tool],
            "tool_choice": {"type": "function", "function": {"name": "cast_vote"}}
        }
    ]

    for case in test_cases:
        print(f"\n\nTest Case: {case['name']}")
        print("-" * 60)

        try:
            kwargs = {
                "model": "anthropic/claude-3.7-sonnet",
                "messages": case["messages"],
                "temperature": 0.5,
                "max_tokens": 150
            }

            if case.get("tools"):
                kwargs["tools"] = case["tools"]
            if case.get("tool_choice"):
                kwargs["tool_choice"] = case["tool_choice"]

            print("\nRequest payload:")
            print(json.dumps(kwargs, indent=2))
            print("\nSending request...")

            response = client.chat.completions.create(**kwargs)
            
            print("\nSuccess! Response:")
            print(f"Model: {response.model}")
            msg = response.choices[0].message
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                print("Tool calls:")
                for tc in msg.tool_calls:
                    print(f"- Name: {tc.function.name}")
                    print(f"  Arguments: {tc.function.arguments}")
            else:
                print(f"Content: {msg.content}")
            
        except Exception as e:
            print(f"\nError occurred:")
            print(f"Type: {type(e).__name__}")
            print(f"Message: {str(e)}")
            print(f"Full error: {repr(e)}")

if __name__ == "__main__":
    test_tool_formats()
