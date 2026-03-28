
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print("Key loaded:", OPENAI_API_KEY[:5] + "*****")
MODEL = "gpt-4o-mini"   # Or "gpt-4-turbo"
