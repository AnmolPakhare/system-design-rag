"""Fetch and extract readable text from external reference links.

Handles HTML (via BeautifulSoup) and PDF (via pypdf). Returns clean text plus a
title, or None if the URL is unreachable, not text, or yields nothing useful.
Known JS/login-walled media hosts are skipped because they carry no article text.
"""
import io
import re
from typing import Optional, Tuple
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

# Hosts that return login walls or JS app shells instead of readable text.
SKIP_HOSTS = (
    "youtube.com", "youtu.be", "twitter.com", "x.com", "facebook.com",
    "instagram.com", "linkedin.com", "t.me", "discord.gg", "discord.com",
    "spotify.com", "tiktok.com",
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

MAX_CHARS = 60_000  # cap per page so a few giant docs don't dominate the index
_WS = re.compile(r"\n{3,}")


def should_skip(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return any(h in host for h in SKIP_HOSTS)


def _html_to_text(html: str) -> Tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer",
                     "form", "svg", "button"]):
        tag.decompose()
    main = soup.find("article") or soup.find("main") or soup.body or soup
    text = main.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines()]
    text = "\n".join(ln for ln in lines if ln)
    return title, _WS.sub("\n\n", text)


def _pdf_to_text(data: bytes) -> Tuple[str, str]:
    reader = PdfReader(io.BytesIO(data))
    title = ""
    try:
        if reader.metadata and reader.metadata.title:
            title = str(reader.metadata.title)
    except Exception:  # noqa: BLE001
        pass
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:  # noqa: BLE001
            continue
    return title, _WS.sub("\n\n", "\n".join(parts))


def fetch(url: str, timeout: int = 20) -> Optional[Tuple[str, str]]:
    """Return (title, text) for a URL, or None if nothing usable was extracted."""
    if should_skip(url):
        return None
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout, stream=True)
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "").lower()
        is_pdf = "application/pdf" in ctype or url.lower().endswith(".pdf")
        content = resp.content
    except Exception:  # noqa: BLE001 - skip any unreachable / erroring URL
        return None

    try:
        if is_pdf:
            title, text = _pdf_to_text(content)
        elif "html" in ctype or "text" in ctype or not ctype:
            title, text = _html_to_text(content.decode("utf-8", errors="ignore"))
        else:
            return None
    except Exception:  # noqa: BLE001
        return None

    text = text.strip()
    if len(text) < 200:  # too little to be worth a chunk
        return None
    return (title or urlparse(url).netloc), text[:MAX_CHARS]
