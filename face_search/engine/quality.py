"""Gate de qualidade do recorte facial.

Antes de codificar um rosto, vale conferir se o recorte tem qualidade
suficiente. Um rosto borrado, escuro demais, estourado ou pequeno demais
geraria um embedding ruim. O gate mede tres grandezas simples e devolve um
veredito, sem descartar o rosto por conta propria — quem chama decide o que
fazer com o resultado.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from face_search.engine.config import EngineConfig
from face_search.engine.detection import DetectedFace


@dataclass(frozen=True)
class QualityReport:
    """Resultado do gate de qualidade de um recorte facial."""

    ok: bool           # True se o recorte passou em todos os criterios
    focus: float       # nitidez, medida pela variancia do Laplaciano
    brightness: float  # brilho medio do recorte, de 0 a 255
    size: int          # menor lado da caixa do rosto, em pixels


def quality_gate(
    aligned_112: np.ndarray,
    face: DetectedFace,
    config: EngineConfig,
) -> QualityReport:
    """Mede foco, brilho e tamanho do rosto e devolve um veredito.

    - Foco: variancia do Laplaciano da imagem em tons de cinza. Quanto maior,
      mais nitida a imagem; valores baixos indicam borrao.
    - Brilho: media dos tons de cinza. Muito baixo = escuro; muito alto =
      estourado.
    - Tamanho: menor lado da caixa do rosto, em pixels da imagem original.
    """

    gray = cv2.cvtColor(aligned_112, cv2.COLOR_BGR2GRAY)
    focus = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())

    x1, y1, x2, y2 = face.bbox
    size = min(x2 - x1, y2 - y1)

    ok = (
        size >= config.min_face
        and focus >= config.min_focus
        and config.min_brightness <= brightness <= config.max_brightness
    )
    return QualityReport(ok=ok, focus=focus, brightness=brightness, size=int(size))
