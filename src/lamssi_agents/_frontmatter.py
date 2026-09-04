"""Parse YAML frontmatter for feature-owned Markdown definitions."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Tuple

log = logging.getLogger("Frontmatter")

_RE = re.compile(r"\A\s*---\s*\n(.*?)\n---\s*\n?(.*)", re.DOTALL)


def parse_frontmatter(text: str) -> Tuple[Dict[str, Any], str]:
    """Split ``text`` into ``(metadata_dict, body_str)``.

    No frontmatter returns ``({}, text)``. Malformed YAML returns empty
    metadata plus a warning log; the body is still returned.
    """
    m = _RE.match(text)
    if not m:
        return {}, text
    try:
        import yaml

        meta = yaml.safe_load(m.group(1)) or {}
        if not isinstance(meta, dict):
            meta = {}
    except Exception as exc:
        log.warning("Bad YAML frontmatter: %s", exc)
        meta = {}
    return meta, m.group(2).strip()


def dump_frontmatter(meta: Dict[str, Any], body: str) -> str:
    """Serialise ``(meta, body)`` back to the Markdown frontmatter form.

    Uses ``yaml.safe_dump`` so a value containing a colon, ``#``, a newline or a
    leading YAML metacharacter survives the round-trip through
    :func:`parse_frontmatter`: a hand-built ``key: value`` string does not.
    """
    import yaml

    front = yaml.safe_dump(
        dict(meta), sort_keys=False, allow_unicode=True, default_flow_style=False
    )
    return f"---\n{front}---\n\n{body.strip()}\n"
