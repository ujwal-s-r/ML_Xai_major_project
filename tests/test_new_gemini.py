"""
Test the new google-genai SDK
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai

# Load environment variables
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ENV_PATH)

api_key = os.getenv("GEMINI_API_KEY")
print(f"API Key found: {bool(api_key)}")


client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=[
        {
            "role": "user",
            "parts": [
                {"text": "Say hello in one sentence."}
            ]
        }
    ]
)

print("RESPONSE:")
print(response.text)