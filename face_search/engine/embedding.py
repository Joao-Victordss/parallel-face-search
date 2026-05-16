"""Codificacao de rostos em vetores faciais (embeddings) com o ArcFace.

Um embedding e um vetor de 512 numeros que representa a identidade de um
rosto. Rostos da mesma pessoa geram vetores proximos; de pessoas diferentes,
vetores distantes. Todos os vetores sao normalizados (norma 1), o que permite
comparar identidades pela similaridade de cosseno.
"""

from __future__ import annotations

import cv2
import numpy as np

from face_search.engine.config import EngineConfig
from face_search.engine.detection import get_face_app


# Identificacao do modelo de codificacao. Gravada no manifesto para que o
# pipeline da webcam recuse embeddings gerados por um modelo incompativel.
EMBEDDING_MODEL = "insightface/arcface-buffalo_l-512d"

# Numero de dimensoes do vetor facial produzido pelo ArcFace.
EMBEDDING_DIM = 512


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    """Normaliza o vetor para norma 1 (norma L2).

    Com vetores normalizados, o produto interno entre dois deles e exatamente
    a similaridade de cosseno.
    """

    norm = np.linalg.norm(vector)
    if norm <= 0:
        return vector
    return vector / norm


def embed_aligned(aligned_112: np.ndarray, config: EngineConfig) -> np.ndarray:
    """Codifica um recorte alinhado de 112x112 em um vetor 512d normalizado."""

    app = get_face_app(config)
    rec_model = app.models["recognition"]
    feat = rec_model.get_feat(aligned_112)
    vector = np.asarray(feat, dtype=np.float32).reshape(-1)
    return _l2_normalize(vector)


def embed_with_tta(aligned_112: np.ndarray, config: EngineConfig) -> np.ndarray:
    """Codifica com test-time augmentation leve (TTA).

    Alem do recorte original, codifica tambem a sua imagem espelhada e tira a
    media dos dois vetores. Isso reduz um pouco o ruido do embedding e custa
    apenas uma inferencia extra. Usado na galeria, onde a qualidade do vetor
    importa mais que a velocidade.
    """

    direct = embed_aligned(aligned_112, config)
    flipped = embed_aligned(cv2.flip(aligned_112, 1), config)
    return _l2_normalize(direct + flipped)


def embed_batch(
    aligned_regions: list[np.ndarray],
    config: EngineConfig,
) -> list[np.ndarray]:
    """Codifica varios recortes 112x112 em uma unica inferencia ArcFace.

    Mais rapido que chamar ``embed_aligned`` em sequencia, porque o ArcFace
    processa todas as imagens em lote. Usado na webcam para codificar as tres
    regioes do rosto (full, upper, periocular) de uma so vez. Devolve um vetor
    512d normalizado por recorte, na mesma ordem da entrada.
    """

    if not aligned_regions:
        return []
    app = get_face_app(config)
    rec_model = app.models["recognition"]
    feats = np.asarray(rec_model.get_feat(list(aligned_regions)), dtype=np.float32)
    if feats.ndim == 1:
        feats = feats.reshape(1, -1)
    return [_l2_normalize(feats[index]) for index in range(feats.shape[0])]
