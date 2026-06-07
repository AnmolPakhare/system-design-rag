"""Extract external reference links (papers, blogs, docs) from the tutorials.

Scans every markdown file in both cloned repos and returns the unique set of
external http(s) URLs worth ingesting, filtering out non-content noise such as
image badges, shields, and asset files.
"""
import re
from typing import Dict, List

from config import REPOS, REPOS_DIR

URL_RE = re.compile(r"https?://[^\s)\]<>\"'`|]+")

# Substrings that mark a URL as noise rather than readable content.
SKIP_DOMAIN_SUBSTR = (
    "shields.io",
    "img.shields",
    "travis-ci",
    "/badge",
    "badgen",
    "githubusercontent.com",
    "gitter.im",
    "opencollective",
    "patreon",
    "paypal",
    "donorbox",
    "translate.google",
    "web.archive.org/web/",  # avoid huge archive snapshots
)

SKIP_EXT = (
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp",
    ".css", ".js", ".ico", ".mp4", ".zip",
)

# Links back into the tutorials' own GitHub repos (and translation forks) just
# re-ingest, as noisy HTML, content we already have as clean markdown. Skip them.
SELF_REPO_SUBSTR = (
    "github.com/donnemartin/system-design-primer",
    "system-design-primer",            # catches translation/mirror forks
    "github.com/karanpratapsingh/system-design",
)


def is_self_repo(url: str) -> bool:
    low = url.lower()
    return "github.com" in low and any(s in low for s in SELF_REPO_SUBSTR)


def _clean(url: str) -> str:
    # Strip trailing markdown/sentence punctuation that the regex may capture.
    return url.rstrip(".,;:!?)'\"")


def extract_links() -> Dict[str, List[str]]:
    """Return {url: [repo names it appeared in]} for content-bearing URLs."""
    urls: Dict[str, List[str]] = {}
    for repo in REPOS:
        root = REPOS_DIR / repo["name"]
        for md in root.rglob("*.md"):
            text = md.read_text(encoding="utf-8", errors="ignore")
            for raw in URL_RE.findall(text):
                url = _clean(raw)
                low = url.lower()
                if any(s in low for s in SKIP_DOMAIN_SUBSTR):
                    continue
                if low.endswith(SKIP_EXT):
                    continue
                if is_self_repo(url):
                    continue
                urls.setdefault(url, [])
                if repo["name"] not in urls[url]:
                    urls[url].append(repo["name"])
    return urls


if __name__ == "__main__":
    links = extract_links()
    print(f"unique content links: {len(links)}")
    from collections import Counter
    from urllib.parse import urlparse

    dom = Counter(urlparse(u).netloc for u in links)
    for d, c in dom.most_common(25):
        print(f"  {c:4d}  {d}")
