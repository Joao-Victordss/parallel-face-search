"""Comando ``face-search-webcam``: busca facial pela webcam.

Compara os rostos capturados pela webcam contra a galeria de procurados, nos
modos sequencial, paralelo ou benchmark. Esta camada cuida apenas dos
argumentos e da validacao; o laco de reconhecimento vive em
``face_search.webcam.pipeline``.
"""

from __future__ import annotations

import argparse
import os

from face_search.webcam import list_available_cameras, run_webcam


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compara rostos da webcam contra os procurados do manifesto."
    )

    # --- Galeria e modo de execucao ----------------------------------------
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Caminho para manifest.json local. Se omitido, carrega do R2.",
    )
    parser.add_argument(
        "--mode",
        choices=("sequential", "parallel", "benchmark"),
        default="parallel",
        help="Modo de comparacao.",
    )
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)

    # --- Camera ------------------------------------------------------------
    parser.add_argument(
        "--camera",
        type=int,
        default=-1,
        help="Indice da camera. Use -1 para tentar automaticamente.",
    )
    parser.add_argument(
        "--max-camera-index",
        type=int,
        default=5,
        help="Maior indice testado no modo automatico e no --list-cameras.",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="Lista cameras disponiveis e encerra.",
    )

    # --- Comparacao e carga do benchmark -----------------------------------
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--frame-scale", type=float, default=0.5)

    # --- Motor de deteccao e codificacao -----------------------------------
    parser.add_argument("--onnx-provider", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--det-size", type=int, default=640)
    parser.add_argument("--min-face", type=int, default=40)

    # --- Thresholds por caminho --------------------------------------------
    # Valores iniciais nao calibrados; ajuste com face-search-evaluate.
    parser.add_argument("--threshold-full", type=float, default=0.38)
    parser.add_argument("--threshold-upper", type=float, default=0.30)
    parser.add_argument("--threshold-periocular", type=float, default=0.24)
    parser.add_argument("--ratio-gap", type=float, default=0.04)

    # --- Tracker -----------------------------------------------------------
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--max-misses", type=int, default=15)

    # --- Re-identificacao de um rosto que sai e volta ----------------------
    parser.add_argument("--reid-threshold", type=float, default=0.5)
    parser.add_argument("--reid-max-age", type=int, default=150)

    # --- Acumulo de evidencia ----------------------------------------------
    parser.add_argument("--decay", type=float, default=0.92)
    parser.add_argument("--confidence-k", type=float, default=3.0)
    parser.add_argument("--confirm-threshold", type=float, default=0.6)
    parser.add_argument("--min-frames", type=int, default=8)
    parser.add_argument("--ref-focus", type=float, default=120.0)

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Confere os limites dos argumentos antes de iniciar o pipeline."""

    if args.frame_scale <= 0 or args.frame_scale > 1:
        raise ValueError("--frame-scale deve ficar entre 0 e 1")
    if args.repeat < 1:
        raise ValueError("--repeat deve ser maior ou igual a 1")
    if args.top_k < 1:
        raise ValueError("--top-k deve ser maior ou igual a 1")
    if args.workers < 1:
        raise ValueError("--workers deve ser maior ou igual a 1")
    if args.max_camera_index < 0:
        raise ValueError("--max-camera-index deve ser maior ou igual a 0")
    if args.det_size < 64:
        raise ValueError("--det-size deve ser maior ou igual a 64")
    if args.min_frames < 1:
        raise ValueError("--min-frames deve ser maior ou igual a 1")
    if not 0.0 < args.decay <= 1.0:
        raise ValueError("--decay deve ficar entre 0 e 1")
    if not -1.0 <= args.reid_threshold <= 1.0:
        raise ValueError("--reid-threshold deve ficar entre -1 e 1")
    if args.reid_max_age < 0:
        raise ValueError("--reid-max-age deve ser maior ou igual a 0")
    for name in ("threshold_full", "threshold_upper", "threshold_periocular"):
        if not -1.0 <= getattr(args, name) <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} deve ficar entre -1 e 1")


def main() -> int:
    from pathlib import Path

    args = parse_args()

    # --list-cameras: so lista e encerra, sem abrir o pipeline.
    if args.list_cameras:
        cameras = list_available_cameras(args.max_camera_index)
        if cameras:
            print("cameras disponiveis:", ", ".join(str(i) for i in cameras))
        else:
            print("nenhuma camera encontrada")
        return 0

    validate_args(args)
    # Converte o caminho do manifesto para Path apenas se foi informado.
    args.manifest = Path(args.manifest) if args.manifest else None
    run_webcam(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
