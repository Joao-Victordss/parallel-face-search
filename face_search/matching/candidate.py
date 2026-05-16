"""Candidatos da galeria e carregamento do manifesto.

Um ``Candidate`` e um procurado pronto para comparacao: traz as tres
variantes de embedding ja convertidas em tuplas (a forma que a comparacao em
Python puro espera). Este modulo tambem carrega o manifesto — de um arquivo
local ou do R2 — e o valida.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from face_search.engine import EMBEDDING_DIM
from face_search.gallery.r2 import GalleryConfig, object_key, r2_client


# As tres variantes de embedding, na ordem canonica usada em todo o projeto.
VARIANTS = ("full", "upper", "periocular")

# Os tres caminhos de comparacao. Cada par associa um rotulo legivel (para a
# tela e os logs) a variante de galeria correspondente. As tres comparacoes
# somam evidencia a uma unica confianca por rosto.
MATCH_METHODS = (
    ("olhos", "periocular"),
    ("face coberta", "upper"),
    ("face exposta", "full"),
)


@dataclass(frozen=True)
class Candidate:
    """Um procurado da galeria, com as tres variantes de embedding."""

    record_id: str
    name: str
    state: str
    source_url: str
    vector_full: tuple[float, ...]
    vector_upper: tuple[float, ...]
    vector_periocular: tuple[float, ...]

    def vector_for(self, variant: str) -> tuple[float, ...]:
        """Devolve o vetor da variante pedida (``full``/``upper``/``periocular``)."""

        return getattr(self, f"vector_{variant}")


@dataclass(frozen=True)
class Match:
    """Resultado de uma comparacao: um candidato e o quanto ele casou."""

    record_id: str
    name: str
    state: str
    source_url: str
    similarity: float  # similaridade de cosseno bruta
    score: float       # similaridade reescalada para exibicao, em [0, 1]


def load_manifest_from_path(path: Path) -> dict[str, Any]:
    """Carrega o manifesto a partir de um arquivo JSON local."""

    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest_from_r2() -> dict[str, Any]:
    """Carrega o manifesto direto do Cloudflare R2, usando as variaveis de ambiente."""

    config = GalleryConfig.from_env(limit=None, no_upload=False)
    client = r2_client(config)
    response = client.get_object(
        Bucket=config.r2_bucket,
        Key=object_key(config, "manifest.json"),
    )
    return json.loads(response["Body"].read().decode("utf-8"))


def load_candidates(manifest: dict[str, Any]) -> list[Candidate]:
    """Converte os registros do manifesto em ``Candidate``s validos.

    Registros em formato antigo (sem as tres variantes de embedding, ou com
    dimensao errada) sao ignorados, com um aviso. Se nada restar, levanta
    ``RuntimeError`` — sinal de que o sync precisa ser rodado de novo.
    """

    candidates: list[Candidate] = []
    skipped = 0

    for record in manifest.get("records", []):
        embeddings = record.get("embeddings")

        # Precisa ter exatamente as tres variantes esperadas.
        if not isinstance(embeddings, dict) or set(embeddings) != set(VARIANTS):
            skipped += 1
            continue

        # Cada variante precisa ser um vetor da dimensao certa.
        if not all(
            isinstance(embeddings[variant], list)
            and len(embeddings[variant]) == EMBEDDING_DIM
            for variant in VARIANTS
        ):
            skipped += 1
            continue

        candidates.append(
            Candidate(
                record_id=record["id"],
                name=record["name"],
                state=record.get("state") or "",
                source_url=record["source_url"],
                vector_full=tuple(float(v) for v in embeddings["full"]),
                vector_upper=tuple(float(v) for v in embeddings["upper"]),
                vector_periocular=tuple(float(v) for v in embeddings["periocular"]),
            )
        )

    if skipped:
        print(
            f"aviso: {skipped} registros ignorados por estarem em formato antigo. "
            "Rode o sync novamente para gerar os embeddings de 512d."
        )
    if not candidates:
        raise RuntimeError(
            "nenhum registro valido no manifesto. Rode o sync para gerar o "
            "novo formato com embeddings full/upper/periocular."
        )
    return candidates
