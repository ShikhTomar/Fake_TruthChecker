import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
key = os.getenv("GEMINI_API_KEY")
print("Key loaded:", bool(key), "| length:", len(key or ""), "| starts with:", (key or "")[:3])

try:
    client = genai.Client(api_key=key)
    r = client.models.generate_content(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        contents="Say hello",
    )
    print("SUCCESS:", r.text)
except Exception as e:
    print("ERROR TYPE:", type(e).__name__)
    print("ERROR MSG:", str(e)[:400])