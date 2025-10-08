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

# ---------------------------------------------------------------------------
# Feature Flags & Minimal Mode
# ---------------------------------------------------------------------------
# We introduce a MINIMAL_MODE to keep the app simple when you only have core keys.
# Core keys considered: GROQ (LLM), HuggingFace token (embeddings), GitHub (optional but present).
# Optional enhancement keys: Judge0, StackOverflow, Google Translate.

OPTIONAL_KEYS_PRESENT = any([
	JUDGE0_API_KEY,
	STACKOVERFLOW_KEY,
	GOOGLE_TRANSLATE_KEY,
])

# Explicit override via env; if not set, auto-compute.
ENV_MINIMAL = os.getenv("MINIMAL_MODE")
if ENV_MINIMAL is not None:
	MINIMAL_MODE = ENV_MINIMAL.lower() in ("1", "true", "yes", "on")
else:
	# Auto minimal if NO optional enhancement keys provided.
	MINIMAL_MODE = not OPTIONAL_KEYS_PRESENT

# Base feature toggles (raw capability)
_RAW_ENABLE_CODE_EXECUTION = bool(JUDGE0_API_KEY)
_RAW_ENABLE_STACKOVERFLOW_SEARCH = bool(STACKOVERFLOW_KEY)
_RAW_ENABLE_GITHUB_SEARCH = bool(GITHUB_TOKEN)
_RAW_ENABLE_TRANSLATION = bool(GOOGLE_TRANSLATE_KEY)

# If in minimal mode, suppress non-core features regardless of raw availability.
if MINIMAL_MODE:
	ENABLE_CODE_EXECUTION = False
	ENABLE_STACKOVERFLOW_SEARCH = False
	ENABLE_TRANSLATION = False
	# GitHub search can still be noisy; keep it but allow disabling if env says so
	ENABLE_GITHUB_SEARCH = _RAW_ENABLE_GITHUB_SEARCH
else:
	ENABLE_CODE_EXECUTION = _RAW_ENABLE_CODE_EXECUTION
	ENABLE_STACKOVERFLOW_SEARCH = _RAW_ENABLE_STACKOVERFLOW_SEARCH
	ENABLE_GITHUB_SEARCH = _RAW_ENABLE_GITHUB_SEARCH
	ENABLE_TRANSLATION = _RAW_ENABLE_TRANSLATION

ACTIVE_FEATURES = [
	feat for feat, enabled in [
		("code_execution", ENABLE_CODE_EXECUTION),
		("stackoverflow_search", ENABLE_STACKOVERFLOW_SEARCH),
		("github_search", ENABLE_GITHUB_SEARCH),
		("translation", ENABLE_TRANSLATION),
	] if enabled
]

def feature_summary() -> str:
	base = f"MINIMAL_MODE={'on' if MINIMAL_MODE else 'off'}\n"
	base += "Active Features: " + (", ".join(ACTIVE_FEATURES) if ACTIVE_FEATURES else "(core only)")
	return base
