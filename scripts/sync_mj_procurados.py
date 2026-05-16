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
import requests
from bs4 import BeautifulSoup
from botocore.exceptions import ClientError
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


LIST_URL = (
    "https://www.gov.br/mj/pt-br/assuntos/sua-seguranca/seguranca-publica/"
    "operacoes-integradas/projeto-captura/lista-de-procurados"
)
USER_AGENT = "mj-procurados-sync/1.0 (+academic dataset sync)"
DATE_RE = re.compile(r"(?P<date>\d{2}/\d{2}/\d{4})")
UPDATED_RE = re.compile(r"Atualizado em\s+(\d{2}/\d{2}/\d{4}\s+\d{2}h\d{2})")
LIST_ITEM_RE = re.compile(r"^(?P<title>.+?)\s+(?P<date>\d{2}/\d{2}/\d{4})$")
FACE_VECTOR_MODEL = "face_recognition/dlib-resnet-v1-128d"
DEFAULT_R2_PREFIX = "mj-procurados"
DEFAULT_OUTPUT_DIR = Path("out")


@dataclass(frozen=True)
class Config:
    limit: int | None
    no_upload: bool
    r2_account_id: str | None
    r2_access_key_id: str | None
    r2_secret_access_key: str | None
    r2_bucket: str | None
    r2_prefix: str
    output_dir: Path
    scrape_delay_seconds: float
    face_detection_model: str

    @classmethod
    def from_env(cls, limit: int | None, no_upload: bool) -> "Config":
        return cls(
            limit=limit,
            no_upload=no_upload,
            r2_account_id=os.getenv("R2_ACCOUNT_ID"),
            r2_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
            r2_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"),
            r2_bucket=os.getenv("R2_BUCKET"),
            r2_prefix=os.getenv("R2_PREFIX", DEFAULT_R2_PREFIX).strip("/") or DEFAULT_R2_PREFIX,
            output_dir=Path(os.getenv("OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR))),
            scrape_delay_seconds=float(os.getenv("SCRAPE_DELAY_SECONDS", "0.5")),
            face_detection_model=os.getenv("FACE_DETECTION_MODEL", "hog"),
        )

    @property
    def upload_enabled(self) -> bool:
        return not self.no_upload

    def require_r2(self) -> None:
        missing = [
            name
            for name, value in {
                "R2_ACCOUNT_ID": self.r2_account_id,
                "R2_ACCESS_KEY_ID": self.r2_access_key_id,
                "R2_SECRET_ACCESS_KEY": self.r2_secret_access_key,
                "R2_BUCKET": self.r2_bucket,
            }.items()
            if not value
        ]
        if missing:
            names = ", ".join(missing)
            raise RuntimeError(f"variaveis R2 ausentes: {names}. Use --no-upload para teste local.")


@dataclass(frozen=True)
class ListedPerson:
    record_id: str
    name: str
    state: str
    listed_date: str | None
    source_url: str


@dataclass
class SyncStats:
    generated_count: int = 0
    reused_count: int = 0


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def fetch(session: requests.Session, url: str) -> requests.Response:
    response = session.get(url, timeout=45)
    response.raise_for_status()
    return response


def build_session() -> requests.Session:
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        status=4,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


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
    import face_recognition

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


def r2_client(config: Config) -> Any | None:
    if not config.upload_enabled:
        return None

    config.require_r2()
    return boto3.client(
        "s3",
        endpoint_url=f"https://{config.r2_account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=config.r2_access_key_id,
        aws_secret_access_key=config.r2_secret_access_key,
        region_name="auto",
    )


def object_key(config: Config, *parts: str) -> str:
    prefix = config.r2_prefix
    clean_parts = [part.strip("/") for part in parts if part]
    return "/".join([prefix, *clean_parts])


def load_existing_manifest(config: Config, client: Any | None) -> dict[str, Any] | None:
    key = object_key(config, "manifest.json")
    if client is None:
        path = config.output_dir / key
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    try:
        response = client.get_object(Bucket=config.r2_bucket, Key=key)
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


def put_json(config: Config, client: Any | None, key: str, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    if client is None:
        path = config.output_dir / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        return

    client.put_object(
        Bucket=config.r2_bucket,
        Key=key,
        Body=body,
        ContentType="application/json; charset=utf-8",
    )


def collect_listed_people(
    session: requests.Session,
    limit: int | None,
    delay_seconds: float,
) -> list[ListedPerson]:
    page_url: str | None = LIST_URL
    people: list[ListedPerson] = []
    seen_pages: set[str] = set()
    seen_records: set[str] = set()

    while page_url and page_url not in seen_pages:
        seen_pages.add(page_url)
        response = fetch(session, page_url)
        page_people, next_url = parse_listing(response.text, page_url)
        new_people = [person for person in page_people if person.record_id not in seen_records]
        people.extend(new_people)
        seen_records.update(person.record_id for person in new_people)
        print(f"coletados {len(new_people)} registros de {page_url}")

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


def can_reuse_embedding(
    existing: dict[str, Any] | None,
    person: ListedPerson,
    updated_at: str | None,
) -> bool:
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
        "face_vector_model": FACE_VECTOR_MODEL,
        "source_image_sha256": image_hash,
        "synced_at": now_iso(),
    }
    return record, True


def sync(limit: int | None, no_upload: bool) -> None:
    config = Config.from_env(limit=limit, no_upload=no_upload)
    session = build_session()
    client = r2_client(config)
    if client is None:
        print(f"upload desativado; gravando em {config.output_dir}/")

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
                detection_model=config.face_detection_model,
            )
            records.append(record)
            if generated:
                stats.generated_count += 1
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
            "nenhum registro valido processado; abortando para nao sobrescrever o manifesto"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sincroniza procurados do MJSP para Cloudflare R2."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limita a quantidade de registros processados.",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Grava em out/ em vez de enviar ao R2.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sync(limit=args.limit, no_upload=args.no_upload)
