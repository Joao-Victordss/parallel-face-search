"""Comando ``face-search-sync``: sincroniza a galeria de procurados.

Faz o scraping da lista publica do MJSP, gera as tres variantes de embedding
de cada procurado e grava o manifesto no Cloudflare R2 (ou em ``out/`` com
``--no-upload``).
"""

from __future__ import annotations

import argparse

from face_search.gallery import sync


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


def main() -> int:
    args = parse_args()
    sync(limit=args.limit, no_upload=args.no_upload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
