"""Settings from environment / .env. Nothing else lives here."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# first model with quota left wins (free tier caps requests per model per day)
GEN_MODELS = os.getenv("GEMINI_GEN_MODELS", "gemini-3.8-flash,gemini-3.6-flash,gemini-3.5-flash-lite").split(",")
BULK_MODELS = os.getenv("GEMINI_BULK_MODELS", "gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-3.6-flash").split(",")
GEMINI_MODEL = GEN_MODELS[0]
DB_PATH = Path(os.getenv("PAPERMIND_DB", ROOT / "data" / "papermind.db"))
DATA_DIR = Path(__file__).resolve().parent / "data"
SYLLABUS_DIR = DATA_DIR / "syllabus"
MODEL_PATH = ROOT / "data" / "topic_model.joblib"
