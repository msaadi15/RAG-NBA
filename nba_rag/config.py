import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
# Keep every download inside this project, including Hugging Face's cache.
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
LLM_ID = os.getenv("NBA_LLM", "Qwen/Qwen2.5-0.5B-Instruct")
EMBED_ID = "sentence-transformers/all-MiniLM-L6-v2"
SOURCES = [
    ("Regular-season MVP winners", "https://www.nba.com/news/history-mvp-award-winners"),
    ("NBA Finals MVP winners", "https://www.nba.com/news/history-finals-mvp-winners"),
    ("NBA champions", "https://www.nba.com/news/history-nba-champions"),
    ("Defensive Player of the Year winners", "https://www.nba.com/news/history-defensive-player-of-the-year-winners"),
    ("Rookie of the Year winners", "https://www.nba.com/news/history-rookie-of-the-year-winners"),
]
