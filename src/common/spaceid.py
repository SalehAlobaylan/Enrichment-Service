"""Immutable vector-space identity for the Embedding & Model Lifecycle System.

MUST stay byte-identical to the Go implementation at
Content-Management-System/src/spaceid/spaceid.go. The canonical serialization is
compact, sorted-key JSON with no ASCII escaping:

    json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

which matches Go's json.Encoder with SetEscapeHTML(false). Golden fixtures pin
cross-language agreement (see tests). A mismatch here silently makes every
vector this service writes uncomparable to what CMS expects, so do not change the
field set or serialization without updating the Go side and the fixtures.

Two identities:
  - space_id:    may these vectors be compared? SHA-256 over the basis contract.
  - producer_id: must this surface be recomputed? SHA-256 over space_id + recipe.
"""
from __future__ import annotations

import hashlib
import json

# Producer-recipe version constants owned by this service's writers. These MUST
# match the Go constants in Content-Management-System/src/spaceid/recipes.go.
RECIPE_CONTENT_TEXT = "content-title-excerpt-body:v1"


def _canonical(obj: dict) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compute_space_id(
    model: str,
    revision: str,
    dimensions: int,
    normalized: bool,
    pooling: str,
) -> str:
    """Return the 64-char hex space_id, or "" when revision is unresolved.

    An unresolved (empty) revision must never masquerade as a stable identity —
    the caller treats "" as "not lifecycle-ready" and does not stamp writes.
    """
    if not revision or not revision.strip():
        return ""
    return _sha256_hex(
        _canonical(
            {
                "model": model,
                "revision": revision,
                "dimensions": dimensions,
                "normalized": normalized,
                "pooling": pooling,
            }
        )
    )


def compute_producer_id(space_id: str, recipe: str) -> str:
    """Return the 64-char hex producer_id, or "" when space_id is empty."""
    if not space_id or not space_id.strip():
        return ""
    return _sha256_hex(_canonical({"recipe": recipe, "space_id": space_id}))
