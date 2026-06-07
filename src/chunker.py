"""Load and chunk markdown documents from the cloned tutorials.

Strategy: split each markdown file at its headings so chunks stay
topically coherent, then window any oversized section into overlapping
character windows. The nearest enclosing heading path is prepended to
every chunk so retrieved snippets keep their context.
"""
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

from config import CHUNK_OVERLAP, CHUNK_SIZE

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Chunk:
    text: str
    source: str  # repo name
    path: str    # file path relative to the repo
    heading: str  # heading trail for context


def _split_by_headings(text: str):
    """Yield (heading_trail, body) blocks split on markdown headings."""
    lines = text.splitlines()
    trail: List[str] = []
    buf: List[str] = []
    current_trail = ""

    def flush():
        body = "\n".join(buf).strip()
        if body:
            yield_blocks.append((current_trail, body))

    yield_blocks = []
    for line in lines:
        m = HEADING_RE.match(line)
        if m:
            # Close the previous block before starting a new heading section.
            flush()
            buf = []
            level = len(m.group(1))
            title = m.group(2).strip()
            trail = trail[: level - 1]
            while len(trail) < level - 1:
                trail.append("")
            trail.append(title)
            current_trail = " > ".join(t for t in trail if t)
        else:
            buf.append(line)
    flush()
    return yield_blocks


def _window(text: str) -> List[str]:
    """Break a long string into overlapping character windows."""
    if len(text) <= CHUNK_SIZE:
        return [text]
    windows = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        windows.append(text[start:end])
        if end >= len(text):
            break
        start = end - CHUNK_OVERLAP
    return windows


def chunk_file(file_path: Path, repo_name: str, repo_root: Path) -> List[Chunk]:
    """Pack consecutive heading-sections into ~CHUNK_SIZE chunks.

    Small sections are merged together so we emit fewer, denser chunks (better
    retrieval context and far fewer embedding calls); any single section larger
    than CHUNK_SIZE is windowed with overlap.
    """
    raw = file_path.read_text(encoding="utf-8", errors="ignore")
    rel = str(file_path.relative_to(repo_root)).replace("\\", "/")
    chunks: List[Chunk] = []

    buf_parts: List[str] = []
    buf_len = 0
    buf_heading = ""

    def flush():
        nonlocal buf_parts, buf_len, buf_heading
        if not buf_parts:
            return
        body = "\n\n".join(buf_parts).strip()
        if body:
            prefix = f"[{repo_name} | {rel}]"
            if buf_heading:
                prefix += f" {buf_heading}"
            chunks.append(
                Chunk(text=f"{prefix}\n\n{body}", source=repo_name, path=rel, heading=buf_heading)
            )
        buf_parts, buf_len, buf_heading = [], 0, ""

    for heading, body in _split_by_headings(raw):
        section = f"{heading}\n{body}" if heading else body

        if len(section) > CHUNK_SIZE:
            # Large section: flush whatever's buffered, then window this one.
            flush()
            for window in _window(section):
                prefix = f"[{repo_name} | {rel}]"
                if heading:
                    prefix += f" {heading}"
                chunks.append(
                    Chunk(text=f"{prefix}\n\n{window}", source=repo_name, path=rel, heading=heading)
                )
            continue

        if buf_len + len(section) > CHUNK_SIZE:
            flush()
        if not buf_heading:
            buf_heading = heading
        buf_parts.append(section)
        buf_len += len(section)

    flush()
    return chunks


def load_repo_chunks(repo_root: Path, repo_name: str) -> List[Chunk]:
    """Walk every markdown file in a repo (all sub-documents / sublinks)."""
    chunks: List[Chunk] = []
    for md in sorted(repo_root.rglob("*.md")):
        # Skip non-content noise.
        if any(part in {".git", "node_modules"} for part in md.parts):
            continue
        chunks.extend(chunk_file(md, repo_name, repo_root))
    return chunks
