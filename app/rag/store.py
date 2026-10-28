"""Knowledge-base plumbing: seed records (source of truth), embeddings, and the Chroma index.

The seed CSVs / policy.md are the authoritative records; Chroma is the *retrieval index*
over them. We compute embeddings ourselves and pass them to Chroma, so the index is
independent of Chroma's embedding-function API and the embedder is recorded per collection
(a mismatch forces a rebuild instead of silently querying with the wrong vector space).

EMBEDDING_PROVIDER=none disables the vector index entirely; retrieval then degrades to a
lexical scan over all records (fine for a KB this size, and what the offline tests use).
"""
from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

VENDORS = "approved_vendors"
CATALOG = "product_catalog"
POLICY = "procurement_policy"


# ------------------------------------------------------------------ seed records
@dataclass
class Vendor:
    vendor_id: str
    name: str
    aliases: list[str]
    tax_id: str
    status: str
    payment_terms: str
    contact_email: str = ""
    notes: str = ""
    learned_aliases: list[str] = field(default_factory=list)

    @property
    def all_names(self) -> list[str]:
        return [self.name, *self.aliases, *self.learned_aliases]


@dataclass
class Product:
    sku: str
    description: str
    category: str
    unit: str
    standard_unit_price: float
    preferred_vendor_id: str = ""


@dataclass
class PolicyChunk:
    chunk_id: str
    heading: str
    text: str


def load_vendors(seed_dir: Path) -> list[Vendor]:
    with open(seed_dir / "vendors.csv", newline="", encoding="utf-8") as fh:
        return [
            Vendor(
                vendor_id=r["vendor_id"], name=r["name"],
                aliases=[a.strip() for a in r.get("aliases", "").split("|") if a.strip()],
                tax_id=r.get("tax_id", ""), status=r.get("status", "active").lower(),
                payment_terms=r.get("payment_terms", ""), contact_email=r.get("contact_email", ""),
                notes=r.get("notes", ""),
            )
            for r in csv.DictReader(fh)
        ]


def load_catalog(seed_dir: Path) -> list[Product]:
    with open(seed_dir / "catalog.csv", newline="", encoding="utf-8") as fh:
        return [
            Product(sku=r["sku"].upper(), description=r["description"], category=r.get("category", ""),
                    unit=r.get("unit", "EA").upper(), standard_unit_price=float(r["standard_unit_price"]),
                    preferred_vendor_id=r.get("preferred_vendor_id", ""))
            for r in csv.DictReader(fh)
        ]


def load_policy_chunks(seed_dir: Path) -> list[PolicyChunk]:
    """Chunk policy.md on '## ' headings - each clause is one retrievable unit."""
    text = (seed_dir / "policy.md").read_text(encoding="utf-8")
    chunks, heading, buf = [], "Preamble", []
    for line in text.splitlines():
        if line.startswith("## "):
            if "".join(buf).strip():
                chunks.append((heading, "\n".join(buf).strip()))
            heading, buf = line[3:].strip(), []
        elif not line.startswith("# "):
            buf.append(line)
    if "".join(buf).strip():
        chunks.append((heading, "\n".join(buf).strip()))
    return [PolicyChunk(f"policy-{i:02d}", h, t) for i, (h, t) in enumerate(chunks)]


# ------------------------------------------------------------------ embeddings
class Embedder:
    name: str = "none"

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class LocalEmbedder(Embedder):
    """fastembed (ONNX) - BAAI/bge-small-en-v1.5 locally, no API key, no torch."""

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        self.name = f"fastembed:{model_name}"
        self._model = TextEmbedding(model_name=model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed(texts)]


class OpenAIEmbedder(Embedder):
    def __init__(self, model_name: str, api_key: str):
        import openai

        self.name = f"openai:{model_name}"
        self._model = model_name
        self._client = openai.OpenAI(api_key=api_key or None)

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self._model, input=texts)
        return [d.embedding for d in resp.data]


@lru_cache(maxsize=4)
def _embedder_for(provider: str, local_model: str, openai_model: str, openai_key: str) -> Optional[Embedder]:
    if provider == "none":
        return None
    if provider == "openai":
        return OpenAIEmbedder(openai_model, openai_key)
    return LocalEmbedder(local_model)


def get_embedder(settings: Settings) -> Optional[Embedder]:
    return _embedder_for(settings.embedding_provider, settings.local_embedding_model,
                         settings.openai_embedding_model, settings.openai_api_key)


# ------------------------------------------------------------------ vector index
class VectorIndex:
    def __init__(self, settings: Settings):
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self.settings = settings
        self.embedder = get_embedder(settings)
        settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(settings.chroma_dir), settings=ChromaSettings(anonymized_telemetry=False)
        )

    def collection(self, name: str):
        return self.client.get_or_create_collection(
            name=name, embedding_function=None,
            metadata={"hnsw:space": "cosine", "embedder": self.embedder.name},
        )

    def is_built(self) -> bool:
        try:
            for name in (VENDORS, CATALOG, POLICY):
                col = self.client.get_collection(name)
                if col.count() == 0 or (col.metadata or {}).get("embedder") != self.embedder.name:
                    return False
            return True
        except Exception:
            return False

    def reset(self, name: str) -> None:
        try:
            self.client.delete_collection(name)
        except Exception:
            pass

    def upsert(self, name: str, ids: list[str], documents: list[str], metadatas: list[dict]) -> None:
        col = self.collection(name)
        for start in range(0, len(ids), 64):
            sl = slice(start, start + 64)
            col.upsert(ids=ids[sl], documents=documents[sl], metadatas=metadatas[sl],
                       embeddings=self.embedder.embed(documents[sl]))

    def query(self, name: str, text: str, k: int) -> list[dict]:
        col = self.collection(name)
        n = min(k, col.count())
        if n == 0:
            return []
        res = col.query(query_embeddings=self.embedder.embed([text]), n_results=n)
        out = []
        for i in range(len(res["ids"][0])):
            out.append({
                "id": res["ids"][0][i],
                "document": res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
                "similarity": round(1 - float(res["distances"][0][i]), 4),
            })
        return out

    def existing_ids(self, name: str, ids: list[str]) -> set[str]:
        if not ids:
            return set()
        return set(self.collection(name).get(ids=ids)["ids"])


_index: Optional[VectorIndex] = None


def get_index(settings: Settings | None = None) -> Optional[VectorIndex]:
    """None when embeddings are disabled or the index can't be opened (callers degrade to lexical)."""
    global _index
    settings = settings or get_settings()
    if settings.embedding_provider == "none":
        return None
    if _index is None or _index.settings is not settings:
        try:
            _index = VectorIndex(settings)
        except Exception as exc:
            logger.error("Vector index unavailable; degrading to lexical retrieval", extra={"error": str(exc)})
            return None
    return _index


def reset_index_cache() -> None:
    global _index
    _index = None


# ------------------------------------------------------------------ text utils
_LEGAL_SUFFIXES = r"\b(incorporated|inc|llc|l\.l\.c|ltd|limited|pvt|private|corp|corporation|co|company|gmbh|plc|sa|the)\b"


def normalize_name(name: str) -> str:
    s = name.lower().replace("&", " and ")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(_LEGAL_SUFFIXES, " ", s)
    return " ".join(s.split())


def normalize_tax_id(tax_id: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (tax_id or "").upper())
