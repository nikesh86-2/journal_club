"""
Journal Club Configuration

Central place for path defaults and small config helpers. Everything here is
overridable via environment variables so the repo has no machine-specific
hardcoded paths.
"""

import os
import re
from pathlib import Path

# Cache directory for memory and other persistent data
_CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
CACHE_DIR = _CACHE_DIR

# Default settings
DEFAULT_TOPICS = []
DEFAULT_DOMAIN = "general"

# Single source of truth for the memory path.
DEFAULT_MEMORY_PATH = os.getenv(
    "JOURNAL_CLUB_LITERATURE_MEMORY_PATH",
    str(_CACHE_DIR / "journal_club_memory.db"),
)

# Default FAISS index path.
DEFAULT_FAISS_INDEX_PATH = os.getenv(
    "JOURNAL_CLUB_FAISS_INDEX_PATH",
    str(_CACHE_DIR / "faiss_index"),
)

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def resolve_env(value):
    """Recursively resolve ${VAR} and ${VAR:-default} placeholders in strings,
    dicts, and lists (used for YAML configs loaded via PyYAML).
    """
    if isinstance(value, str):
        def _sub(match):
            var = match.group(1)
            resolved = os.getenv(var)
            if resolved is None or resolved == "":
                resolved = match.group(2)
            return resolved if resolved is not None else match.group(0)

        return _ENV_PATTERN.sub(_sub, value)
    if isinstance(value, dict):
        return {key: resolve_env(val) for key, val in value.items()}
    if isinstance(value, list):
        return [resolve_env(val) for val in value]
    return value


def load_topics(config_path: str | None = None) -> list:
    """Load topics from config/topics.yaml."""
    try:
        import yaml
    except ImportError:
        return []

    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config" / "topics.yaml"

    if not Path(config_path).exists():
        return []

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            return data.get("topics", []) or []
    except Exception:
        return []


def get_topic_by_name(name: str) -> dict | None:
    """Get the config entry for a topic by name."""
    for topic in load_topics():
        if topic.get("name") == name:
            return topic
    return None
