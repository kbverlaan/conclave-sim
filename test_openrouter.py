import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def test_openrouter():
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

    # Test message
    messages = [
        {"role": "user", "content": "Say hello!"}
    ]

    try:
        print("Sending test request to OpenRouter...")
        response = client.chat.completions.create(
            model="anthropic/claude-3.7-sonnet",
            messages=messages,
            max_tokens=100,
            temperature=0.7
        )
        
        print("\nSuccess! Response received:")
        print(f"Model: {response.model}")
        print(f"Message: {response.choices[0].message.content}")
        
    except Exception as e:
        print(f"\nError occurred:")
        print(f"Type: {type(e).__name__}")
        print(f"Message: {str(e)}")

if __name__ == "__main__":
    test_openrouter()
