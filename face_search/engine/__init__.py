"""Motor de deteccao e codificacao facial.

Subpacote que concentra tudo relacionado a transformar uma imagem em vetores
faciais comparaveis. O sync (galeria) e a webcam (consulta) usam este mesmo
motor, garantindo que os dois lados sejam processados de forma identica.

Modulos:

- ``config``    — ``EngineConfig``, os parametros do motor.
- ``detection`` — deteccao e alinhamento de rostos (SCRFD).
- ``embedding`` — codificacao de rostos em vetores 512d (ArcFace).
- ``quality``   — gate de qualidade do recorte facial.
- ``regions``   — extracao das tres regioes do rosto (full/upper/periocular).

Este ``__init__`` re-exporta a API publica do motor para que o restante do
codigo possa importar tudo direto de ``face_search.engine``.
"""

from __future__ import annotations

from face_search.engine.config import EngineConfig
from face_search.engine.detection import (
    DetectedFace,
    align_crop,
    detect_faces,
    get_face_app,
    largest_face,
)
from face_search.engine.embedding import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    embed_aligned,
    embed_batch,
    embed_with_tta,
)
from face_search.engine.quality import QualityReport, quality_gate
from face_search.engine.regions import (
    ALIGNED_SIZE,
    NEUTRAL_FILL,
    REGION_EXTRACTORS,
    region_full,
    region_periocular,
    region_upper,
)

__all__ = [
    "EngineConfig",
    "DetectedFace",
    "align_crop",
    "detect_faces",
    "get_face_app",
    "largest_face",
    "EMBEDDING_DIM",
    "EMBEDDING_MODEL",
    "embed_aligned",
    "embed_batch",
    "embed_with_tta",
    "QualityReport",
    "quality_gate",
    "ALIGNED_SIZE",
    "NEUTRAL_FILL",
    "REGION_EXTRACTORS",
    "region_full",
    "region_periocular",
    "region_upper",
]
