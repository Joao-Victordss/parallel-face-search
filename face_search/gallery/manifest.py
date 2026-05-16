"""Estrutura e validacao do manifesto da galeria.

O manifesto (``manifest.json``) e o arquivo central da galeria: lista todos
os procurados, cada um com suas tres variantes de embedding e metadados de
proveniencia. Este modulo reune as funcoes que entendem esse formato — desde
a validacao de um registro ate a decisao de reaproveitar um embedding ja
gerado.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from face_search.engine import EMBEDDING_DIM, EMBEDDING_MODEL, REGION_EXTRACTORS
from face_search.gallery.scraper import ListedPerson


# Identifica o metodo de mascara sintetica usado na galeria. Gravado no
# registro para que o cache so reaproveite embeddings gerados pelo mesmo
# metodo de neutralizacao de regiao.
SYNTHETIC_MASK_MODEL = "region-neutralize@1"


def now_iso() -> str:
    """Data e hora atuais em UTC, no formato ISO 8601 com segundos."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def existing_records_by_id(manifest: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Indexa os registros de um manifesto anterior pelo seu id.

    Devolve um dicionario vazio quando nao ha manifesto. Serve para localizar
    rapidamente o registro antigo de um procurado durante o sync.
    """

    if not manifest:
        return {}
    return {
        record.get("id"): record
        for record in manifest.get("records", [])
        if isinstance(record, dict) and record.get("id")
    }


def has_valid_embeddings(record: dict[str, Any] | None) -> bool:
    """Confere se o registro tem as tres variantes de embedding completas.

    Um registro valido precisa do campo ``embeddings`` com exatamente as
    chaves ``full``, ``upper`` e ``periocular``, cada uma com um vetor de
    ``EMBEDDING_DIM`` numeros.
    """

    if not record:
        return False
    embeddings = record.get("embeddings")
    if not isinstance(embeddings, dict):
        return False
    if set(embeddings) != set(REGION_EXTRACTORS):
        return False
    return all(
        isinstance(embeddings[variant], list)
        and len(embeddings[variant]) == EMBEDDING_DIM
        for variant in embeddings
    )


def can_reuse_embedding(
    existing: dict[str, Any] | None,
    person: ListedPerson,
    updated_at: str | None,
) -> bool:
    """Decide se o embedding de um sync anterior pode ser reaproveitado.

    O reaproveitamento so e seguro quando nada que influencia o embedding
    mudou: a URL de origem, a data de atualizacao da pagina, a data de
    listagem, o modelo de codificacao e o metodo de mascara sintetica.
    """

    if not has_valid_embeddings(existing):
        return False
    assert existing is not None
    return (
        existing.get("source_url") == person.source_url
        and existing.get("updated_at") == updated_at
        and existing.get("listed_date") == person.listed_date
        and existing.get("embedding_model") == EMBEDDING_MODEL
        and existing.get("embedding_dim") == EMBEDDING_DIM
        and existing.get("synthetic_mask_model") == SYNTHETIC_MASK_MODEL
    )
