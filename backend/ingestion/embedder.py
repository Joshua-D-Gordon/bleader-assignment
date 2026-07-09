"""OpenAI embeddings (text-embedding-3-large, 3072-dim).

Batches inputs to stay within request limits. Used for both ingestion (embedding
`chunk.embedding_text`) and query time (embedding the user's question).
"""
from __future__ import annotations

from ..app import config
from ..app.clients import openai_client

_BATCH = 100  # inputs per request


def embed_texts(texts: list[str]) -> list[list[float]]:
    client = openai_client()
    out: list[list[float]] = []
    for i in range(0, len(texts), _BATCH):
        batch = texts[i:i + _BATCH]
        resp = client.embeddings.create(
            model=config.EMBEDDING_MODEL,
            input=batch,
            dimensions=config.EMBEDDING_DIM,
        )
        out.extend(d.embedding for d in resp.data)
    return out


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
