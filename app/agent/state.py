import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
api_key_multimodal = os.getenv("GOOGLE_API_KEY_MULTIMODAL")

# Configura variable de entorno para Gemini
os.environ["GOOGLE_API_KEY"] = api_key
