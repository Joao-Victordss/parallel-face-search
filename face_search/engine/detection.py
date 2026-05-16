"""Deteccao e alinhamento de rostos com o detector SCRFD do InsightFace.

Este modulo encontra rostos numa imagem (com ou sem mascara), entrega os
cinco pontos de referencia de cada rosto e recorta o rosto alinhado em
112x112 pixels, pronto para a codificacao pelo ArcFace.

A importacao do InsightFace e cara, por isso ela e feita de forma preguicosa
dentro de ``_load_app``, so quando o motor e usado pela primeira vez.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from face_search.engine.config import EngineConfig
from face_search.engine.regions import ALIGNED_SIZE


@dataclass(frozen=True)
class DetectedFace:
    """Um rosto localizado numa imagem.

    Guarda a caixa delimitadora, os cinco pontos faciais (usados para alinhar
    o recorte) e a confianca da deteccao reportada pelo SCRFD.
    """

    bbox: tuple[int, int, int, int]  # cantos x1, y1, x2, y2
    kps: np.ndarray                  # 5x2: olho esq, olho dir, nariz, boca esq, boca dir
    det_score: float                 # confianca da deteccao, em [0, 1]

    @property
    def area(self) -> int:
        """Area da caixa em pixels. Usada para escolher o maior rosto."""

        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)


@lru_cache(maxsize=4)
def _load_app(provider_key: str, pack: str, det_size: int, ctx_id: int):
    """Carrega e prepara o ``FaceAnalysis`` do InsightFace.

    O resultado e cacheado por configuracao (``lru_cache``): carregar os
    modelos ONNX e lento, entao isso so acontece uma vez por combinacao de
    parametros.
    """

    from insightface.app import FaceAnalysis

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if provider_key == "cuda"
        else ["CPUExecutionProvider"]
    )
    app = FaceAnalysis(
        name=pack,
        # So precisamos da deteccao e do reconhecimento; os demais modelos
        # (atributos, landmarks 3D) sao dispensados para economizar memoria.
        allowed_modules=["detection", "recognition"],
        providers=providers,
    )
    app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))
    return app


def get_face_app(config: EngineConfig):
    """Devolve a instancia ``FaceAnalysis`` preparada para a configuracao."""

    return _load_app(
        config.onnx_provider,
        config.insightface_pack,
        config.det_size,
        config.ctx_id(),
    )


def detect_faces(image_bgr: np.ndarray, config: EngineConfig) -> list[DetectedFace]:
    """Detecta todos os rostos de uma imagem BGR (com e sem mascara)."""

    app = get_face_app(config)
    det_model = app.models["detection"]
    bboxes, kpss = det_model.detect(image_bgr, max_num=0, metric="default")

    faces: list[DetectedFace] = []
    for index in range(bboxes.shape[0]):
        x1, y1, x2, y2, score = bboxes[index]
        faces.append(
            DetectedFace(
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                kps=kpss[index],
                det_score=float(score),
            )
        )
    return faces


def largest_face(faces: list[DetectedFace]) -> DetectedFace:
    """Devolve o maior rosto da lista.

    Usado na foto oficial do procurado, em que se assume que o rosto
    principal e o maior da imagem.
    """

    if not faces:
        raise ValueError("nenhum rosto detectado na imagem")
    return max(faces, key=lambda face: face.area)


def align_crop(image_bgr: np.ndarray, kps: np.ndarray) -> np.ndarray:
    """Recorta e alinha o rosto para 112x112 usando os cinco pontos do SCRFD.

    O alinhamento aplica uma transformacao afim que leva os olhos, o nariz e a
    boca para posicoes fixas. Isso normaliza rotacao e escala, condicao para
    que o ArcFace gere embeddings comparaveis.
    """

    from insightface.utils import face_align

    return face_align.norm_crop(image_bgr, landmark=kps, image_size=ALIGNED_SIZE)
