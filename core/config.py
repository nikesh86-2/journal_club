"""
Journal Club Configuration
"""

from pathlib import Path

# Cache directory for memory and other persistent data
_CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
CACHE_DIR = _CACHE_DIR

# Default settings
DEFAULT_TOPICS = []
DEFAULT_DOMAIN = "general"
