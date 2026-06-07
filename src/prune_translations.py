"""Prune redundant/noisy chunks the judge flagged, without a full reset (so the
useful crawled web/paper chunks are preserved):

  - non-English translated-README chunks (README-ja.md, README-zh-Hans.md, ...)
  - crawled GitHub pages of the tutorials' own repos / translation forks, which
    duplicate the clean markdown we already index

    python src/prune_translations.py
"""
from pathlib import PurePosixPath

import chromadb

from chunker import is_translated_readme
from config import CHROMA_DIR, COLLECTION_NAME
from links import is_self_repo


def main() -> None:
    col = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(COLLECTION_NAME)
    before = col.count()
    got = col.get(include=["metadatas"])

    translated, self_repo = [], []
    for cid, m in zip(got["ids"], got["metadatas"]):
        path = m.get("path", "")
        if m.get("source") == "web":
            if is_self_repo(path):
                self_repo.append(cid)
        elif is_translated_readme(PurePosixPath(path).name):
            translated.append(cid)

    to_delete = translated + self_repo
    print(
        f"collection size: {before}; deleting {len(translated)} translated-README "
        f"+ {len(self_repo)} self-repo web chunks = {len(to_delete)}"
    )
    if to_delete:
        col.delete(ids=to_delete)
    print(f"collection size now: {col.count()}")


if __name__ == "__main__":
    main()
