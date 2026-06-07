"""Central configuration loaded from the .env file."""
import os
from pathlib import Path

from dotenv import load_dotenv

# Project root = parent of the src/ directory.
ROOT = Path(__file__).resolve().parent.parent

load_dotenv(ROOT / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")

# Which engine produces the vectors used for retrieval:
#   "local"  -> Chroma's bundled all-MiniLM-L6-v2 (offline, no API, no quota)
#   "gemini" -> gemini-embedding-001 via the API (subject to free-tier quotas)
# Generation always uses Gemini (GEMINI_MODEL) regardless of this setting.
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").lower()

# Where cloned tutorials and the vector DB live.
DATA_DIR = ROOT / "data"
REPOS_DIR = DATA_DIR / "repos"
CHROMA_DIR = DATA_DIR / "chroma"
COLLECTION_NAME = "system_design"

# The two tutorials this RAG system is trained on. Every markdown file in
# each repository (i.e. every "sublink" / sub-document) is ingested.
REPOS = [
    {
        "name": "system-design-primer",
        "url": "https://github.com/donnemartin/system-design-primer.git",
    },
    {
        "name": "system-design",
        "url": "https://github.com/karanpratapsingh/system-design.git",
    },
]

# Chunking parameters (characters). Larger chunks keep the total number of
# embeddings well under the gemini-embedding-001 free-tier daily request cap
# while still fitting comfortably within the model's input token limit.
CHUNK_SIZE = 4000
CHUNK_OVERLAP = 300

# Embedding API batch size. Each item in a batch counts as one request against
# the free-tier limit (100 requests/minute), so we batch then pace.
EMBED_BATCH_SIZE = 50

# Seconds to sleep between embedding batches to stay under 100 requests/minute
# on the free tier (50 items per batch / 32s ~= 94 requests/minute).
EMBED_PACE_SECONDS = 32

# Number of chunks to retrieve per query.
TOP_K = 6

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set. Add it to the .env file.")
