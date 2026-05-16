"""Construcao dos registros da galeria e orquestracao do sync.

Este modulo junta as pecas do subpacote ``gallery``: pega cada procurado
descoberto pelo ``scraper``, baixa a foto oficial, gera as tres variantes de
embedding com o motor facial e grava o resultado via ``r2``. O manifesto
final consolida todos os registros.

A foto oficial e usada apenas em memoria — nunca e gravada no R2 nem em
disco. So os vetores e os metadados sao persistidos.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import requests

from face_search import engine
from face_search.engine import EngineConfig
from face_search.gallery.manifest import (
    SYNTHETIC_MASK_MODEL,
    can_reuse_embedding,
    existing_records_by_id,
    now_iso,
)
from face_search.gallery.r2 import (
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
    fetch,
    image_url_for,
    parse_updated_at,
)


@dataclass
class SyncStats:
    """Contadores de um sync: quantos embeddings foram gerados e reusados."""

    generated_count: int = 0
    reused_count: int = 0


def build_face_embeddings(
    image_bytes: bytes,
    engine_config: EngineConfig,
) -> dict[str, list[float]]:
    """Gera as tres variantes de embedding (full, upper, periocular).

    O rosto da foto oficial e detectado e alinhado uma unica vez. Cada
    variante aplica uma neutralizacao de regiao sobre o mesmo recorte alinhado
    e recodifica, colocando a galeria no mesmo dominio do probe mascarado da
    webcam.
    """

    image = cv2.imdecode(
        np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if image is None:
        raise ValueError("nao foi possivel decodificar a imagem")

    faces = engine.detect_faces(image, engine_config)
    if not faces:
        raise ValueError("nenhum rosto detectado na imagem")

    face = engine.largest_face(faces)
    aligned = engine.align_crop(image, face.kps)

    quality = engine.quality_gate(aligned, face, engine_config)
    if not quality.ok:
        raise ValueError(
            f"qualidade insuficiente (foco={quality.focus:.0f}, "
            f"brilho={quality.brightness:.0f}, tamanho={quality.size}px)"
        )

    embeddings: dict[str, list[float]] = {}
    for variant, extract_region in engine.REGION_EXTRACTORS.items():
        region = extract_region(aligned)
        vector = engine.embed_with_tta(region, engine_config)
        # Arredonda para 8 casas: estabiliza o JSON sem perda relevante de
        # precisao na comparacao por cosseno.
        embeddings[variant] = [round(float(value), 8) for value in vector]
    return embeddings


def build_record(
    session: requests.Session,
    person: ListedPerson,
    existing: dict[str, Any] | None,
    engine_config: EngineConfig,
) -> tuple[dict[str, Any], bool]:
    """Monta o registro de um procurado.

    Devolve um par ``(registro, gerado)``. Quando o embedding do sync anterior
    ainda e valido, o registro e reaproveitado e ``gerado`` vem ``False``.
    Caso contrario, a foto e baixada, codificada e ``gerado`` vem ``True``.
    """

    detail_html = fetch(session, person.source_url).text
    updated_at = parse_updated_at(detail_html)

    # Caminho de cache: nada que afeta o embedding mudou desde o ultimo sync.
    if can_reuse_embedding(existing, person, updated_at):
        reused = dict(existing)
        reused["synced_at"] = now_iso()
        return reused, False

    # Caminho de geracao: baixa a foto e produz os tres embeddings.
    image_url = image_url_for(person.source_url)
    image_bytes = fetch(session, image_url).content
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    embeddings = build_face_embeddings(image_bytes, engine_config)

    record = {
        "id": person.record_id,
        "name": person.name,
        "state": person.state,
        "listed_date": person.listed_date,
        "updated_at": updated_at,
        "source_url": person.source_url,
        "source_image_sha256": image_hash,
        "embeddings": embeddings,
        "embedding_model": engine.EMBEDDING_MODEL,
        "embedding_dim": engine.EMBEDDING_DIM,
        "synthetic_mask_model": SYNTHETIC_MASK_MODEL,
        "synced_at": now_iso(),
    }
    return record, True


def sync(limit: int | None, no_upload: bool) -> None:
    """Executa o pipeline completo de sincronizacao da galeria.

    Passos: carrega a configuracao e o manifesto anterior (cache), coleta a
    lista publica, monta um registro por procurado (gerando ou reaproveitando
    o embedding) e grava o manifesto consolidado.

    Se a lista vier vazia ou nenhum registro for processado com sucesso, o
    sync aborta sem gravar — para nao sobrescrever um manifesto bom por um
    vazio.
    """

    config = GalleryConfig.from_env(limit=limit, no_upload=no_upload)
    engine_config = EngineConfig.from_env()
    session = build_session()
    client = r2_client(config)
    if client is None:
        print(f"upload desativado; gravando em {config.output_dir}/")

    # Manifesto anterior: usado como cache de embeddings.
    existing_manifest = load_existing_manifest(config, client)
    existing_by_id = existing_records_by_id(existing_manifest)

    listed_people = collect_listed_people(
        session,
        limit=config.limit,
        delay_seconds=config.scrape_delay_seconds,
    )
    if not listed_people:
        raise RuntimeError(
            "nenhum registro encontrado; abortando para nao sobrescrever o manifesto"
        )

    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    stats = SyncStats()

    for index, person in enumerate(listed_people, start=1):
        try:
            record, generated = build_record(
                session=session,
                person=person,
                existing=existing_by_id.get(person.record_id),
                engine_config=engine_config,
            )
            records.append(record)
            if generated:
                stats.generated_count += 1
                # Cada registro tambem e gravado individualmente, para
                # consulta pontual sem baixar o manifesto inteiro.
                put_json(
                    config,
                    client,
                    object_key(config, "records", f"{person.record_id}.json"),
                    record,
                )
            else:
                stats.reused_count += 1
            print(f"[{index}/{len(listed_people)}] ok {person.record_id}")
        except Exception as exc:
            failures.append(
                {
                    "id": person.record_id,
                    "name": person.name,
                    "source_url": person.source_url,
                    "error": str(exc),
                }
            )
            print(f"[{index}/{len(listed_people)}] falhou {person.record_id}: {exc}")

        time.sleep(config.scrape_delay_seconds)

    if not records:
        raise RuntimeError(
            "nenhum registro valido processado; abortando para nao sobrescrever "
            "o manifesto"
        )

    manifest = {
        "source": LIST_URL,
        "generated_at": now_iso(),
        "record_count": len(records),
        "failure_count": len(failures),
        "generated_embedding_count": stats.generated_count,
        "reused_embedding_count": stats.reused_count,
        "records": sorted(
            records,
            key=lambda item: (item.get("state") or "", item.get("name") or ""),
        ),
        "failures": failures,
    }
    put_json(config, client, object_key(config, "manifest.json"), manifest)
    print(
        "sync concluido: "
        f"{len(records)} registros, {stats.generated_count} embeddings novos, "
        f"{stats.reused_count} reutilizados, {len(failures)} falhas"
    )
