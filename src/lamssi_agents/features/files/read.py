"""The Files feature's scoped text, document, and structured-data reader."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from lamssi_tools import CapabilityContext, Expose, Int, Str, err, tool

from .space import FileSpace, suggest_near_match

log = logging.getLogger(__name__)

_DOCUMENT_SUFFIXES = frozenset({
    ".pdf",
    ".docx", ".dotx",
    ".pptx",
    ".xlsx",
    ".html", ".htm", ".xhtml",
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp",
})
_document_converter = None


def _get_document_converter():
    """Load and cache the optional document converter."""
    global _document_converter
    if _document_converter is None:
        try:
            from docling.document_converter import DocumentConverter
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "docling is required for document extraction: "
                'install with `pip install "lamssi-agents[documents]"`'
            ) from exc
        _document_converter = DocumentConverter()
    return _document_converter


def _document_supported(target: Path) -> bool:
    return target.suffix.lower() in _DOCUMENT_SUFFIXES


def _extract_document(
    target: Path,
    *,
    start_page: int = 0,
    end_page: int = 0,
    rel_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract a supported document into a ``read_file`` result."""
    try:
        converter = _get_document_converter()
    except RuntimeError as exc:
        return err(str(exc))

    page_range: Optional[Tuple[int, int]] = None
    if start_page or end_page:
        start = max(start_page, 1) if start_page else 1
        end = end_page if end_page else 10_000
        if start > end:
            return err(f"Empty page range: start={start_page}, end={end_page}")
        page_range = (start, end)

    try:
        result = converter.convert(
            str(target), **({"page_range": page_range} if page_range else {})
        )
    except PermissionError:
        return err(f"Permission denied: {target}")
    except TypeError:
        try:
            result = converter.convert(str(target))
        except Exception as exc:
            return err(f"Document conversion failed: {exc}")
    except Exception as exc:
        return err(f"Document conversion failed: {exc}")

    document = getattr(result, "document", None)
    if document is None:
        return err(f"Document conversion returned no document: {target.name}")
    try:
        content = document.export_to_markdown()
    except Exception as exc:
        return err(f"Markdown export failed: {exc}")

    pages_total = _page_count(document)
    if page_range is not None and pages_total:
        pages_extracted = max(0, min(page_range[1], pages_total) - page_range[0] + 1)
    else:
        pages_extracted = pages_total or 1
    return {
        "content": content,
        "pages_extracted": pages_extracted,
        "pages_total": pages_total or 1,
        "path": rel_path if rel_path is not None else str(target),
    }


def _page_count(document: Any) -> int:
    """Return a best-effort page count across docling versions."""
    pages = getattr(document, "pages", None)
    if pages is None:
        return 0
    try:
        return len(pages)
    except TypeError:
        try:
            return sum(1 for _ in pages)
        except Exception:
            return 0


@tool(
    group="files",
    # A file's own cap, not the generic 8,000-char output net: most source files exceed it.
    truncation=50_000,
    truncation_hint=(
        "You have the start of the file, not all of it. Re-call with start_line set "
        "past what you were given, or search for the part you need first."
    ),
    expose=Expose.ALL,
    inject_context=True,
    description=(
        "Read one file, whatever it holds. Use this once you know which file you want; "
        "if you are still looking for it, use fs first: reading files to find "
        "something is slow and fills this conversation with text you did not need. "
        "Text comes back as text. Spreadsheets, CSVs and array files come back as a "
        "STRUCTURED summary: columns, dtypes, statistics and a sample: rather than "
        "thousands of rows, and carry a 'kind' key. Documents come back as extracted "
        "text. You do not choose which: the file decides. "
        "The path is relative to the workspace, or absolute; an absolute path outside "
        "the workspace is read only with the user's approval. For paginated documents, "
        "start_line and end_line are 1-based page numbers; asking for a line range "
        "always returns text, so use it to see a data file's raw rows."
    ),
    # Safe in the workspace and the readable dirs; a read outside either prompts.
    approval="conditional",
    guard_role="recovery",
    keywords="open show view display cat contents inspect look",
    parameters={
        "path": Str("Path relative to the workspace, or absolute."),
        "start_line": Int(
            "1-based inclusive. 0 = start. Forces a text read.", ge=0
        ),
        "end_line": Int("1-based inclusive. 0 = end. Forces a text read.", ge=0),
        "sheet": Str("Sheet name or 0-based index, for spreadsheets."),
        "max_rows": Int("Cap rows read from a data file (0 = all).", ge=0),
        "sample_rows": Int(
            "Rows of a data file to include verbatim.", ge=0, le=100
        ),
    },
)
def read_file(
    ctx: CapabilityContext,
    path: str = "",
    start_line: int = 0,
    end_line: int = 0,
    sheet: str = "0",
    max_rows: int = 0,
    sample_rows: int = 5,
    **kw,
) -> Dict:
    if not path:
        return err("path is required", retriable=False)
    space = ctx.require(FileSpace)

    route = space.resolve(path, allow_external=True)
    if route.error is not None:
        return route.error
    target = route.target
    if target is None:
        return err(
            f"Path {path!r} could not be resolved.",
            hint="Give a path relative to the workspace, or an absolute path.",
            retriable=False,
        )

    if not target.is_file():
        wanted = Path(path).name
        hint_parts = [
            f"Use fs(command='find . -name {wanted}') to find it anywhere in the "
            "workspace: the directory you named may be the part that is wrong"
        ]
        base = route.base
        if base is not None:
            suggestion = suggest_near_match(target, base)
            if suggestion:
                hint_parts.insert(0, f"Did you mean {suggestion!r}?")
        return err(
            f"File not found: {path}",
            hint=" ".join(hint_parts) + ".",
            retriable=False,
        )

    display_path = route.display(target)

    # Route structured formats through their extractor.
    from .table import STRUCTURED_BY_DEFAULT, extract_table

    if _document_supported(target) and target.suffix.lower() not in STRUCTURED_BY_DEFAULT:
        return _extract_document(
            target, start_page=start_line, end_page=end_line, rel_path=display_path
        )

    # A line range is an explicit request for lines, so it wins over the summary.
    wants_lines = bool(start_line or end_line)
    if not wants_lines and target.suffix.lower() in STRUCTURED_BY_DEFAULT:
        summary = extract_table(
            target,
            sheet=sheet,
            max_rows=max_rows,
            sample_rows=sample_rows,
            rel_path=display_path,
        )
        if "error" not in summary:
            return summary
        _table_error = summary.get("error", "")
    else:
        _table_error = ""

    try:
        text = target.read_text("utf-8")
    except UnicodeDecodeError:
        if _table_error:
            # It is a data file we could not parse, not prose we could not decode.
            return err(_table_error, retriable=False)
        return err(
            f"Cannot read binary file: {path}",
            hint="This file is binary: skip it, or use a tool that extracts text.",
            retriable=False,
        )
    except PermissionError:
        return err(f"Permission denied: {path}", retriable=False)
    except Exception as e:
        return err(f"Read error: {e}", retriable=False)

    lines = text.splitlines(keepends=True)
    total = len(lines)
    if start_line or end_line:
        s = max(start_line - 1, 0) if start_line else 0
        e = end_line if end_line else total
        text = "".join(lines[s:e])

    out: Dict = {
        "content": text,
        "lines": total,
        "path": display_path,
        "root": route.root,
    }
    return out
