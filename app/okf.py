"""
OKF (Open Knowledge Format) loader.

Parses OKF entries — Markdown files with a YAML frontmatter block — into
LangChain Documents whose metadata carries the structured fields (id, category,
confidence, source). This metadata is what enables filtered retrieval later
(e.g. restrict to confidence: high, or a given category) rather than relying on
semantic similarity alone.

OKF is a content format that complements RAG: consistently structured, atomic,
well-tagged entries make retrieval more precise. Ref: Google's Open Knowledge
Format (2026).
"""

import os
import glob
import yaml
from langchain_core.documents import Document


def parse_okf(text):
    """Split an OKF entry into (metadata_dict, body). Frontmatter is a YAML
    block delimited by leading '---' lines."""
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            meta = yaml.safe_load(parts[1]) or {}
            return meta, parts[2].strip()
    return {}, text.strip()


def load_okf_docs(okf_dir="data/okf"):
    """Load all OKF entries in a directory into Documents with structured
    metadata. Missing fields default to empty strings (Chroma metadata cannot
    be None)."""
    docs = []
    for path in sorted(glob.glob(os.path.join(okf_dir, "*.md"))):
        with open(path, encoding="utf-8") as f:
            meta, body = parse_okf(f.read())
        docs.append(Document(
            page_content=body,
            metadata={
                "source": os.path.basename(path),
                "okf_id": str(meta.get("id", "")),
                "category": str(meta.get("category", "")),
                "confidence": str(meta.get("confidence", "")),
                "okf_source": str(meta.get("source", "")),
            },
        ))
    return docs