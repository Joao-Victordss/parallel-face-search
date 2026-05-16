"""Medidas de similaridade entre vetores faciais.

A comparacao de identidades usa similaridade de cosseno: dois embeddings da
mesma pessoa apontam para direcoes parecidas (cosseno alto); de pessoas
diferentes, direcoes distantes (cosseno baixo).
"""

from __future__ import annotations

import math


def cosine_similarity(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Similaridade de cosseno entre dois vetores, escrita em Python puro.

    IMPORTANTE — esta funcao e mantida sem NumPy de proposito. Ela e a carga
    de trabalho que torna visivel a diferenca entre os modos sequencial e
    paralelo no benchmark. Uma versao vetorizada com NumPy daria o mesmo
    resultado, apenas mais rapido — e isso apagaria a demonstracao de
    paralelismo do projeto.

    Os vetores ja chegam normalizados; a divisao pelas normas e mantida por
    robustez e clareza.
    """

    dot = sum(left * right for left, right in zip(a, b))
    norm_a = math.sqrt(sum(value * value for value in a))
    norm_b = math.sqrt(sum(value * value for value in b))
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def similarity_to_score(similarity: float, threshold: float) -> float:
    """Converte uma similaridade de cosseno em um score de exibicao em [0, 1].

    Reescala a faixa ``[threshold, 1.0]`` para ``[0.0, 1.0]``: uma
    similaridade exatamente no limiar vale 0, e a similaridade maxima vale 1.
    Valores abaixo do limiar sao zerados.
    """

    if threshold >= 1.0:
        return 0.0
    return max(0.0, min(1.0, (similarity - threshold) / (1.0 - threshold)))
