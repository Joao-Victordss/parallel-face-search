from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import boto3
import face_recognition
import requests
from bs4 import BeautifulSoup
from botocore.exceptions import ClientError


LIST_URL = (
    "https://www.gov.br/mj/pt-br/assuntos/sua-seguranca/seguranca-publica/"
    "operacoes-integradas/projeto-captura/lista-de-procurados"
)
USER_AGENT = "mj-procurados-sync/1.0 (+academic dataset sync)"
DATE_RE = re.compile(r"(?P<date>\d{2}/\d{2}/\d{4})")
UPDATED_RE = re.compile(r"Atualizado em\s+(\d{2}/\d{2}/\d{4}\s+\d{2}h\d{2})")
LIST_ITEM_RE = re.compile(r"^(?P<title>.+?)\s+(?P<date>\d{2}/\d{2}/\d{4})$")


@dataclass(frozen=True)
class ListedPerson:
    record_id: str
    name: str
    state: str
    listed_date: str | None
    source_url: str


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def fetch(session: requests.Session, url: str) -> requests.Response:
    response = session.get(url, timeout=45)
    response.raise_for_status()
    return response


def record_id_from_url(url: str) -> str:
    path = urlparse(url).path.rstrip("/")
    if path.endswith("/view"):
        path = path[: -len("/view")]
    return path.rsplit("/", 1)[-1]


def parse_name_state(raw_title: str) -> tuple[str, str]:
    title = normalize_space(DATE_RE.sub("", raw_title))
    match = re.match(r"^(?P<name>.+?)\s*-\s*(?P<state>[A-ZÁÀÂÃÉÊÍÓÔÕÚÜÇ ]+)\.?$", title)
    if not match:
        return title.rstrip(".").strip(), ""

    name = match.group("name").strip().rstrip(".")
    state = match.group("state").strip().rstrip(".")
    return name, state


def parse_listing(html: str, page_url: str) -> tuple[list[ListedPerson], str | None]:
    soup = BeautifulSoup(html, "html.parser")
    records: list[ListedPerson] = []
    seen: set[str] = set()
    next_url: str | None = None

    for anchor in soup.find_all("a", href=True):
        text = normalize_space(anchor.get_text(" "))
        href = urljoin(page_url, anchor["href"])

        if "Próximo" in text:
            next_url = href
            continue

        if "/lista-de-procurados/" not in href or not href.rstrip("/").endswith("/view"):
            continue

        match = LIST_ITEM_RE.match(text)
        if not match:
            continue

        source_url = href
        record_id = record_id_from_url(source_url)
        if record_id in seen:
            continue

        name, state = parse_name_state(match.group("title"))
        records.append(
            ListedPerson(
                record_id=record_id,
                name=name,
                state=state,
                listed_date=match.group("date"),
                source_url=source_url,
            )
        )
        seen.add(record_id)

    return records, next_url


def image_url_for(source_url: str) -> str:
    path = source_url.rstrip("/")
    if path.endswith("/view"):
        path = path[: -len("/view")]
    return f"{path}/@@images/image"


def parse_updated_at(html: str) -> str | None:
    text = normalize_space(BeautifulSoup(html, "html.parser").get_text(" "))
    match = UPDATED_RE.search(text)
    return match.group(1) if match else None


def largest_face_location(locations: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    return max(locations, key=lambda box: (box[2] - box[0]) * (box[1] - box[3]))


def build_face_embedding(image_bytes: bytes, detection_model: str) -> list[float]:
    image = face_recognition.load_image_file(BytesIO(image_bytes))
    locations = face_recognition.face_locations(image, model=detection_model)
    if not locations:
        raise ValueError("nenhum rosto detectado na imagem")

    location = largest_face_location(locations)
    encodings = face_recognition.face_encodings(
        image,
        known_face_locations=[location],
        num_jitters=1,
        model="small",
    )
    if not encodings:
        raise ValueError("nao foi possivel gerar vetor facial")

    return [round(float(value), 8) for value in encodings[0]]


def r2_client() -> Any | None:
    account_id = os.getenv("R2_ACCOUNT_ID")
    access_key = os.getenv("R2_ACCESS_KEY_ID")
    secret_key = os.getenv("R2_SECRET_ACCESS_KEY")
    if not all([account_id, access_key, secret_key, os.getenv("R2_BUCKET")]):
        return None

    return boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
    )


def object_key(*parts: str) -> str:
    prefix = os.getenv("R2_PREFIX", "mj-procurados").strip("/")
    clean_parts = [part.strip("/") for part in parts if part]
    return "/".join([prefix, *clean_parts])


def load_existing_manifest(client: Any | None) -> dict[str, Any] | None:
    if client is None:
        path = Path("out") / object_key("manifest.json")
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    bucket = os.environ["R2_BUCKET"]
    try:
        response = client.get_object(Bucket=bucket, Key=object_key("manifest.json"))
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code")
        if error_code in {"NoSuchKey", "404", "NotFound"}:
            return None
        print(f"manifesto anterior indisponivel, seguindo sem cache: {exc}")
        return None
    except Exception as exc:
        print(f"manifesto anterior indisponivel, seguindo sem cache: {exc}")
        return None

    return json.loads(response["Body"].read().decode("utf-8"))


def put_json(client: Any | None, key: str, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    if client is None:
        path = Path("out") / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        return

    client.put_object(
        Bucket=os.environ["R2_BUCKET"],
        Key=key,
        Body=body,
        ContentType="application/json; charset=utf-8",
    )


def collect_listed_people(session: requests.Session, limit: int | None, delay_seconds: float) -> list[ListedPerson]:
    page_url: str | None = LIST_URL
    people: list[ListedPerson] = []
    seen_pages: set[str] = set()

    while page_url and page_url not in seen_pages:
        seen_pages.add(page_url)
        response = fetch(session, page_url)
        page_people, next_url = parse_listing(response.text, page_url)
        people.extend(page_people)
        print(f"coletados {len(page_people)} registros de {page_url}")

        if limit and len(people) >= limit:
            return people[:limit]

        page_url = next_url
        if page_url:
            time.sleep(delay_seconds)

    return people


def existing_records_by_id(manifest: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not manifest:
        return {}
    return {
        record.get("id"): record
        for record in manifest.get("records", [])
        if isinstance(record, dict) and record.get("id")
    }


def can_reuse_embedding(existing: dict[str, Any] | None, person: ListedPerson, updated_at: str | None) -> bool:
    if not existing:
        return False
    return (
        existing.get("source_url") == person.source_url
        and existing.get("updated_at") == updated_at
        and existing.get("listed_date") == person.listed_date
        and isinstance(existing.get("face_vector"), list)
    )


def build_record(
    session: requests.Session,
    person: ListedPerson,
    existing: dict[str, Any] | None,
    detection_model: str,
) -> tuple[dict[str, Any], bool]:
    detail_html = fetch(session, person.source_url).text
    updated_at = parse_updated_at(detail_html)

    if can_reuse_embedding(existing, person, updated_at):
        reused = dict(existing)
        reused["synced_at"] = now_iso()
        return reused, False

    image_url = image_url_for(person.source_url)
    image_response = fetch(session, image_url)
    image_bytes = image_response.content
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    face_vector = build_face_embedding(image_bytes, detection_model=detection_model)

    record = {
        "id": person.record_id,
        "name": person.name,
        "state": person.state,
        "listed_date": person.listed_date,
        "updated_at": updated_at,
        "source_url": person.source_url,
        "face_vector": face_vector,
        "face_vector_model": "face_recognition/dlib-resnet-v1-128d",
        "source_image_sha256": image_hash,
        "synced_at": now_iso(),
    }
    return record, True


def sync(limit: int | None, no_upload: bool) -> None:
    delay_seconds = float(os.getenv("SCRAPE_DELAY_SECONDS", "0.5"))
    detection_model = os.getenv("FACE_DETECTION_MODEL", "hog")

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    client = None if no_upload else r2_client()
    if client is None:
        print("R2 nao configurado ou upload desativado; gravando em out/")

    existing_manifest = load_existing_manifest(client)
    existing_by_id = existing_records_by_id(existing_manifest)
    listed_people = collect_listed_people(session, limit=limit, delay_seconds=delay_seconds)
    if not listed_people:
        raise RuntimeError("nenhum registro encontrado; abortando para nao sobrescrever o manifesto")

    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    generated_count = 0
    reused_count = 0

    for index, person in enumerate(listed_people, start=1):
        try:
            record, generated = build_record(
                session=session,
                person=person,
                existing=existing_by_id.get(person.record_id),
                detection_model=detection_model,
            )
            records.append(record)
            if generated:
                generated_count += 1
                put_json(client, object_key("records", f"{person.record_id}.json"), record)
            else:
                reused_count += 1
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

        time.sleep(delay_seconds)

    if not records:
        raise RuntimeError("nenhum registro valido processado; abortando para nao sobrescrever o manifesto")

    manifest = {
        "source": LIST_URL,
        "generated_at": now_iso(),
        "record_count": len(records),
        "failure_count": len(failures),
        "generated_embedding_count": generated_count,
        "reused_embedding_count": reused_count,
        "records": sorted(records, key=lambda item: (item.get("state") or "", item.get("name") or "")),
        "failures": failures,
    }
    put_json(client, object_key("manifest.json"), manifest)
    print(
        "sync concluido: "
        f"{len(records)} registros, {generated_count} embeddings novos, "
        f"{reused_count} reutilizados, {len(failures)} falhas"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sincroniza procurados do MJSP para Cloudflare R2.")
    parser.add_argument("--limit", type=int, default=None, help="Limita a quantidade de registros processados.")
    parser.add_argument("--no-upload", action="store_true", help="Grava em out/ em vez de enviar ao R2.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sync(limit=args.limit, no_upload=args.no_upload)
