# config/settings.py

import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
HF_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Define model names
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant" # <-- UPDATE TO THIS MODEL