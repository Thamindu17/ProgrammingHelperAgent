# config/settings.py

import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
HF_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Free API Keys for Enhanced Features
JUDGE0_API_KEY = os.getenv("JUDGE0_API_KEY", "")  # Free: 50 requests/day
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")  # Free: 5000 requests/hour
STACKOVERFLOW_KEY = os.getenv("STACKOVERFLOW_KEY", "")  # Free: 10k requests/day
GOOGLE_TRANSLATE_KEY = os.getenv("GOOGLE_TRANSLATE_KEY", "")  # Free: 500k chars/month
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")  # Free tier available

# Define model names
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant"

# API Endpoints
JUDGE0_ENDPOINT = "https://judge0-ce.p.rapidapi.com"
STACKOVERFLOW_API = "https://api.stackexchange.com/2.3"
GITHUB_API = "https://api.github.com"
DEVDOCS_API = "https://devdocs.io"

# Feature Flags
ENABLE_CODE_EXECUTION = bool(JUDGE0_API_KEY)
ENABLE_STACKOVERFLOW_SEARCH = bool(STACKOVERFLOW_KEY)
ENABLE_GITHUB_SEARCH = bool(GITHUB_TOKEN)
ENABLE_TRANSLATION = bool(GOOGLE_TRANSLATE_KEY) # <-- UPDATE TO THIS MODEL