import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
api_key_multimodal = os.getenv("GOOGLE_API_KEY_MULTIMODAL")

# Configura variable de entorno para Gemini
os.environ["GOOGLE_API_KEY"] = api_key


llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
