"""Configuracao e acesso ao armazenamento Cloudflare R2.

A galeria de vetores faciais e gravada no Cloudflare R2 (um armazenamento de
objetos compativel com a API S3). Este modulo concentra a configuracao do
pipeline de sincronizacao e as funcoes de leitura e escrita de objetos.

Quando o upload esta desativado (``--no-upload``), os mesmos objetos sao
gravados em disco, na pasta ``out/``, com a mesma estrutura de chaves.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError


# Prefixo padrao das chaves de objeto no R2 (uma "pasta" virtual).
DEFAULT_R2_PREFIX = "mj-procurados"

# Pasta local usada quando o upload esta desativado.
DEFAULT_OUTPUT_DIR = Path("out")


@dataclass(frozen=True)
class GalleryConfig:
    """Configuracao do pipeline de sincronizacao da galeria."""

    limit: int | None              # limite opcional de registros a processar
    no_upload: bool                # se True, grava em disco em vez do R2
    r2_account_id: str | None
    r2_access_key_id: str | None
    r2_secret_access_key: str | None
    r2_bucket: str | None
    r2_prefix: str
    output_dir: Path               # pasta local usada quando no_upload=True
    scrape_delay_seconds: float    # pausa entre requisicoes ao site do MJ

    @classmethod
    def from_env(cls, limit: int | None, no_upload: bool) -> "GalleryConfig":
        """Cria a configuracao lendo as credenciais das variaveis de ambiente."""

        return cls(
            limit=limit,
            no_upload=no_upload,
            r2_account_id=os.getenv("R2_ACCOUNT_ID"),
            r2_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
            r2_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"),
            r2_bucket=os.getenv("R2_BUCKET"),
            r2_prefix=os.getenv("R2_PREFIX", DEFAULT_R2_PREFIX).strip("/")
            or DEFAULT_R2_PREFIX,
            output_dir=Path(os.getenv("OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR))),
            scrape_delay_seconds=float(os.getenv("SCRAPE_DELAY_SECONDS", "0.5")),
        )

    @property
    def upload_enabled(self) -> bool:
        return not self.no_upload

    def require_r2(self) -> None:
        """Garante que todas as credenciais do R2 estao presentes.

        Levanta ``RuntimeError`` listando o que falta. Chamado antes de
        qualquer operacao que dependa do R2.
        """

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
            raise RuntimeError(
                f"variaveis R2 ausentes: {names}. Use --no-upload para teste local."
            )


def r2_client(config: GalleryConfig) -> Any | None:
    """Cria o cliente S3 apontado para o endpoint do Cloudflare R2.

    Devolve ``None`` quando o upload esta desativado — nesse caso as funcoes
    de leitura e escrita operam sobre o disco local.
    """

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


def object_key(config: GalleryConfig, *parts: str) -> str:
    """Monta a chave de um objeto, juntando o prefixo configurado e as partes.

    Exemplo: ``object_key(config, "records", "fulano.json")`` ->
    ``"mj-procurados/records/fulano.json"``.
    """

    prefix = config.r2_prefix
    clean_parts = [part.strip("/") for part in parts if part]
    return "/".join([prefix, *clean_parts])


def load_existing_manifest(
    config: GalleryConfig,
    client: Any | None,
) -> dict[str, Any] | None:
    """Le o manifesto da sincronizacao anterior, se existir.

    O manifesto anterior serve de cache: registros cuja pagina publica nao
    mudou tem o embedding reaproveitado, evitando recodificar tudo. Qualquer
    falha de leitura e tratada como "sem cache" — o sync segue normalmente.
    """

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


def put_json(
    config: GalleryConfig,
    client: Any | None,
    key: str,
    payload: dict[str, Any],
) -> None:
    """Grava um dicionario como JSON, no R2 ou em disco.

    A serializacao usa ``sort_keys`` para que o mesmo conteudo gere sempre o
    mesmo arquivo, facilitando comparacoes e cache.
    """

    body = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode(
        "utf-8"
    )

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
