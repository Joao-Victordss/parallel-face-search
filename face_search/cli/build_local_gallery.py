"""Comando ``face-search-build-gallery``: monta uma galeria de imagens locais.

Util para demonstracao e teste: permite rodar a webcam contra rostos
conhecidos sem depender da lista publica do MJ. Cada imagem informada vira um
registro de galeria, com as tres variantes de embedding.

Exemplo:

    face-search-build-gallery \\
        --face "Voce=amostras/EU.jpeg" \\
        --face "Meliante=amostras/meliante.png" \\
        --out out/demo/manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from face_search.engine import EMBEDDING_DIM, EMBEDDING_MODEL, EngineConfig
from face_search.gallery import SYNTHETIC_MASK_MODEL, build_face_embeddings, now_iso


def slugify(name: str) -> str:
    """Converte um nome legivel num identificador simples (so letras e numeros)."""

    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "registro"


def parse_face_arg(values: list[str]) -> list[tuple[str, Path]]:
    """Interpreta os argumentos ``--face NOME=CAMINHO``."""

    faces: list[tuple[str, Path]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"--face espera nome=caminho, recebido: {value}")
        name, raw_path = value.split("=", 1)
        faces.append((name.strip(), Path(raw_path.strip())))
    return faces


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monta um manifest.json de galeria a partir de imagens locais."
    )
    parser.add_argument(
        "--face",
        action="append",
        required=True,
        metavar="NOME=CAMINHO",
        help="Rosto da galeria. Pode repetir.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("out/demo/manifest.json"),
        help="Caminho do manifest.json gerado.",
    )
    parser.add_argument("--onnx-provider", choices=("cuda", "cpu"), default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    faces = parse_face_arg(args.face)
    engine_config = EngineConfig(onnx_provider=args.onnx_provider)

    records = []
    for name, path in faces:
        image_bytes = path.read_bytes()
        embeddings = build_face_embeddings(image_bytes, engine_config)
        records.append(
            {
                "id": slugify(name),
                "name": name,
                "state": "",
                "listed_date": None,
                "updated_at": None,
                "source_url": f"local://{path}",
                "source_image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                "embeddings": embeddings,
                "embedding_model": EMBEDDING_MODEL,
                "embedding_dim": EMBEDDING_DIM,
                "synthetic_mask_model": SYNTHETIC_MASK_MODEL,
                "synced_at": now_iso(),
            }
        )
        print(f"ok {name} ({path.name})")

    manifest = {
        "source": "local",
        "generated_at": now_iso(),
        "record_count": len(records),
        "failure_count": 0,
        "records": records,
        "failures": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"galeria gravada em {args.out} ({len(records)} registros)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
