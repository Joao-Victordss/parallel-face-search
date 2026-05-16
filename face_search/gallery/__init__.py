"""Montagem da galeria de vetores faciais.

Este subpacote constroi a galeria contra a qual a webcam compara os rostos.
O fluxo, do site publico ao manifesto, e:

- ``scraper``  — percorre a lista publica do MJ e descobre os procurados.
- ``builder``  — baixa as fotos, gera os embeddings e orquestra o sync.
- ``manifest`` — formato, validacao e cache do ``manifest.json``.
- ``r2``       — configuracao e acesso ao armazenamento Cloudflare R2.
"""

from __future__ import annotations

from face_search.gallery.builder import (
    SyncStats,
    build_face_embeddings,
    build_record,
    sync,
)
from face_search.gallery.manifest import (
    SYNTHETIC_MASK_MODEL,
    can_reuse_embedding,
    existing_records_by_id,
    has_valid_embeddings,
    now_iso,
)
from face_search.gallery.r2 import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_R2_PREFIX,
    GalleryConfig,
    load_existing_manifest,
    object_key,
    put_json,
    r2_client,
)
from face_search.gallery.scraper import (
    LIST_URL,
    ListedPerson,
    build_session,
    collect_listed_people,
)

__all__ = [
    "SyncStats",
    "build_face_embeddings",
    "build_record",
    "sync",
    "SYNTHETIC_MASK_MODEL",
    "can_reuse_embedding",
    "existing_records_by_id",
    "has_valid_embeddings",
    "now_iso",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_R2_PREFIX",
    "GalleryConfig",
    "load_existing_manifest",
    "object_key",
    "put_json",
    "r2_client",
    "LIST_URL",
    "ListedPerson",
    "build_session",
    "collect_listed_people",
]
