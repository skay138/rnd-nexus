import json
import logging
import os
from pathlib import Path
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)

def _resolve_fixtures_dir() -> Path:
    env = os.environ.get("FIXTURES_DIR")
    if env:
        return Path(env)
    # Docker: mcp-server/src/ → /app/src/, data/ → /app/data/  (4 parents up = /app)
    # Local dev: mcp-server/src/infrastructure/repositories/ → 5 parents up = repo root
    candidate_4 = Path(__file__).parent.parent.parent.parent / "data" / "fixtures"
    candidate_5 = Path(__file__).parent.parent.parent.parent.parent / "data" / "fixtures"
    return candidate_4 if candidate_4.exists() else candidate_5

_DEFAULT_FIXTURES = _resolve_fixtures_dir()

def load_fixture(
    filename: str,
    fixtures_dir: Optional[Path] = None,
) -> Union[list[dict[str, Any]], dict[str, Any]]:
    path = (fixtures_dir or _DEFAULT_FIXTURES) / filename
    data: Union[list[dict[str, Any]], dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    logger.debug("[InMemory] loaded %s (%s records)", filename, len(data) if isinstance(data, list) else "dict")
    return data

def keyword_score(text: str, keywords: list[str]) -> int:
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)
