"""Acumulo de evidencia por rosto rastreado.

A analise de um unico frame e ruidosa, ainda mais com mascara. Por isso o
sistema nao decide a identidade num frame so: cada frame soma evidencia ao
candidato que ele aponta, e a confianca cresce conforme frames consistentes
concordam.

Frames ruidosos (mascara, foco ruim, deteccao fraca) contribuem com peso
menor. Assim o caminho com mascara exige naturalmente mais frames para
atingir a mesma confianca de um rosto exposto.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


# Peso de evidencia por variante de galeria. Quanto maior a cobertura da
# mascara, menos confiavel o frame e menor o peso: o rosto exposto vale
# 1.0, a metade superior 0.6 e a faixa dos olhos apenas 0.35.
MASK_WEIGHTS = {
    "full": 1.0,
    "upper": 0.6,
    "periocular": 0.35,
}


def clamp(value: float, low: float, high: float) -> float:
    """Limita um valor ao intervalo [low, high]."""

    return max(low, min(high, value))


def frame_weight(
    variant: str,
    focus: float,
    ref_focus: float,
    det_score: float,
) -> float:
    """Calcula o peso de um frame para o acumulo de evidencia.

    O peso combina tres fatores, todos em [0, 1]:

    - ``w_mascara``  — depende da variante (ver ``MASK_WEIGHTS``).
    - ``w_qualidade`` — foco do recorte, normalizado por um foco de referencia.
    - ``w_deteccao``  — confianca da deteccao do SCRFD.

    Resultado: ``w_frame = w_mascara * w_qualidade * w_deteccao``.
    """

    w_mask = MASK_WEIGHTS.get(variant, 0.35)
    w_quality = clamp(focus / ref_focus, 0.0, 1.0) if ref_focus > 0 else 0.0
    w_detection = clamp(det_score, 0.0, 1.0)
    return w_mask * w_quality * w_detection


@dataclass
class EvidenceAccumulator:
    """Evidencia acumulada de um track ao longo dos frames.

    Cada track (rosto rastreado) tem um acumulador. Os tres caminhos de
    comparacao somam evidencia a este mesmo objeto, formando uma unica
    confianca por rosto.
    """

    decay: float = 0.92          # esquecimento da evidencia antiga, por frame
    confidence_k: float = 3.0    # constante de saturacao da confianca
    scores: dict[str, float] = field(default_factory=dict)  # evidencia por candidato
    frames_seen: int = 0         # total de frames observados por este track
    last_update_frame: int = -1  # indice do ultimo frame observado

    def observe(self, contributions: list[tuple[str, float]], frame_index: int) -> None:
        """Registra um frame, somando a evidencia dos caminhos de comparacao.

        ``contributions`` traz um par ``(record_id, valor)`` para cada caminho
        de match que encontrou candidato neste frame. O fluxo e:

        1. O decaimento e aplicado uma unica vez ao frame: toda evidencia
           antiga e multiplicada por ``decay``, esquecendo o passado aos poucos.
        2. Cada contribuicao positiva e somada ao seu candidato.

        Uma lista vazia apenas registra o frame e aplica o decaimento, o que
        cobre o caso de um rosto sem nenhum match utilizavel.
        """

        self.frames_seen += 1
        self.last_update_frame = frame_index

        # Decaimento: a evidencia antiga perde forca a cada frame.
        for key in list(self.scores):
            self.scores[key] *= self.decay

        # Soma a evidencia nova de cada caminho de comparacao.
        for record_id, value in contributions:
            if value > 0.0:
                self.scores[record_id] = self.scores.get(record_id, 0.0) + value

    def leading(self) -> tuple[str | None, float]:
        """Devolve o candidato com mais evidencia acumulada e seu score bruto."""

        if not self.scores:
            return None, 0.0
        record_id = max(self.scores, key=lambda key: self.scores[key])
        return record_id, self.scores[record_id]

    def confidence(self) -> float:
        """Confianca do track em [0, 1), saturando suavemente com a evidencia.

        A formula ``1 - exp(-evidencia / k)`` faz a confianca subir rapido no
        inicio e desacelerar perto de 1, sem nunca atingir 100%.
        """

        _, raw = self.leading()
        if raw <= 0.0:
            return 0.0
        return 1.0 - math.exp(-raw / self.confidence_k)

    def is_confident(self, confirm_threshold: float, min_frames: int) -> bool:
        """Indica se o track ja pode ser promovido a candidato confirmado.

        Exige duas condicoes: ter sido observado por um numero minimo de
        frames e ter confianca acima do limiar de confirmacao.
        """

        return (
            self.frames_seen >= min_frames
            and self.confidence() >= confirm_threshold
        )
