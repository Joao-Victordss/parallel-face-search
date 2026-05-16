"""Comando ``face-search-evaluate``: avalia a acuracia do reconhecimento.

Le o manifesto da galeria, roda o harness de avaliacao sintetica e imprime um
resumo. Opcionalmente grava o relatorio completo em JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from face_search.evaluation import build_report, print_summary
from face_search.matching import (
    load_candidates,
    load_manifest_from_path,
    load_manifest_from_r2,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Avalia a acuracia do reconhecimento facial do manifesto."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Caminho para manifest.json local. Se omitido, carrega do R2.",
    )
    parser.add_argument(
        "--max-impostors",
        type=int,
        default=50,
        help="Quantos impostores amostrar por identidade.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Caminho para gravar o relatorio JSON. Opcional.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_impostors < 1:
        raise ValueError("--max-impostors deve ser maior ou igual a 1")

    manifest = (
        load_manifest_from_path(args.manifest)
        if args.manifest
        else load_manifest_from_r2()
    )
    candidates = load_candidates(manifest)
    report = build_report(candidates, args.max_impostors)

    print_summary(report)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"relatorio gravado em {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
