"""Comando ``face-search-match``: testa o reconhecimento com fotos em disco.

Roda o mesmo pipeline da webcam, mas a partir de fotos em disco em vez da
camera. Compara cada probe pelos tres caminhos (olhos, face coberta, face
exposta), sem precisar de webcam nem do scraping da lista.

Sem argumentos, usa as fotos da pasta ``amostras/``: monta uma galeria com
``meliante.png`` e ``EU.jpeg`` e consulta as duas como probe. O esperado e
que cada foto ranqueie a si mesma em primeiro lugar e que a similaridade
entre pessoas diferentes fique baixa.

Uso com fotos proprias:

    face-search-match \\
        --gallery procurado=caminho/foto_oficial.jpg \\
        --probe caminho/frame.jpg
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2

from face_search import engine
from face_search.engine import EngineConfig
from face_search.matching import MATCH_METHODS, cosine_similarity


# Thresholds iniciais, iguais aos defaults da webcam. Nao calibrados.
THRESHOLDS = {"full": 0.38, "upper": 0.30, "periocular": 0.24}

# Raiz do repositorio, para localizar a pasta amostras/ no modo padrao.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GALLERY = {
    "meliante": REPO_ROOT / "amostras" / "meliante.png",
    "eu": REPO_ROOT / "amostras" / "EU.jpeg",
}
DEFAULT_PROBES = [
    REPO_ROOT / "amostras" / "EU.jpeg",
    REPO_ROOT / "amostras" / "meliante.png",
]


@dataclass
class GalleryEntry:
    """Uma entrada da galeria de teste, com as tres variantes de embedding."""

    name: str
    embeddings: dict[str, tuple[float, ...]]


def load_image(path: Path):
    """Le uma imagem do disco, levantando erro claro se ela nao existir."""

    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"nao foi possivel ler a imagem: {path}")
    return image


def detect_and_align(path: Path, engine_config: EngineConfig):
    """Detecta o maior rosto da imagem e devolve o recorte alinhado 112x112."""

    image = load_image(path)
    faces = engine.detect_faces(image, engine_config)
    if not faces:
        raise ValueError(f"nenhum rosto detectado em {path.name}")
    face = engine.largest_face(faces)
    aligned = engine.align_crop(image, face.kps)
    return face, aligned


def build_gallery_entry(
    name: str,
    path: Path,
    engine_config: EngineConfig,
) -> GalleryEntry:
    """Gera as tres variantes de embedding de uma foto de galeria."""

    _, aligned = detect_and_align(path, engine_config)
    embeddings: dict[str, tuple[float, ...]] = {}
    for variant, extract_region in engine.REGION_EXTRACTORS.items():
        region = extract_region(aligned)
        vector = engine.embed_with_tta(region, engine_config)
        embeddings[variant] = tuple(float(value) for value in vector)
    return GalleryEntry(name=name, embeddings=embeddings)


def run_probe(
    path: Path,
    gallery: list[GalleryEntry],
    engine_config: EngineConfig,
) -> None:
    """Compara uma foto de consulta pelos tres caminhos, de forma separada."""

    _, aligned = detect_and_align(path, engine_config)
    print(f"probe: {path.name}")

    for label, variant in MATCH_METHODS:
        threshold = THRESHOLDS[variant]
        region = engine.REGION_EXTRACTORS[variant](aligned)
        probe_vector = tuple(
            float(value) for value in engine.embed_aligned(region, engine_config)
        )
        ranking = sorted(
            (
                (cosine_similarity(probe_vector, entry.embeddings[variant]), entry.name)
                for entry in gallery
            ),
            reverse=True,
        )
        best_similarity, best_name = ranking[0]
        verdict = best_name if best_similarity >= threshold else "Desconhecido"
        print(
            f"  [{label}] -> {verdict}  "
            f"(sim={best_similarity:+.4f}, threshold={threshold})"
        )
        for similarity, name in ranking:
            print(f"      {name:<14} {similarity:+.4f}")
    print()


def parse_gallery_arg(values: list[str]) -> dict[str, Path]:
    """Interpreta os argumentos ``--gallery NOME=CAMINHO``."""

    gallery: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--gallery espera nome=caminho, recebido: {value}")
        name, raw_path = value.split("=", 1)
        gallery[name.strip()] = Path(raw_path.strip())
    return gallery


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Testa o reconhecimento com fotos em disco."
    )
    parser.add_argument(
        "--gallery",
        action="append",
        default=None,
        metavar="NOME=CAMINHO",
        help="Entrada da galeria. Pode repetir. Padrao: pasta amostras/.",
    )
    parser.add_argument(
        "--probe",
        action="append",
        type=Path,
        default=None,
        help="Foto de consulta. Pode repetir. Padrao: pasta amostras/.",
    )
    parser.add_argument("--onnx-provider", choices=("cuda", "cpu"), default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    engine_config = EngineConfig(onnx_provider=args.onnx_provider)

    gallery_paths = (
        parse_gallery_arg(args.gallery) if args.gallery else dict(DEFAULT_GALLERY)
    )
    probe_paths = args.probe if args.probe else list(DEFAULT_PROBES)

    print("montando galeria...")
    gallery = [
        build_gallery_entry(name, path, engine_config)
        for name, path in gallery_paths.items()
    ]
    print(f"galeria: {', '.join(entry.name for entry in gallery)}")
    print()

    for probe_path in probe_paths:
        run_probe(probe_path, gallery, engine_config)

    print(
        "lembrete: similaridade alta entre pessoas diferentes ou baixa entre "
        "a mesma pessoa indica que os thresholds precisam de calibracao com "
        "face-search-evaluate."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
