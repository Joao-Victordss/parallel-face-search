"""Extracao das tres regioes do rosto e geometria do recorte alinhado.

O sistema compara cada rosto por tres caminhos. Em vez de detectar o tipo de
mascara, ele neutraliza (apaga) as regioes que uma mascara cobriria e
recodifica o rosto. Assim a galeria e o probe ficam sempre no mesmo dominio:

- ``full``       — rosto inteiro, sem neutralizacao (caminho "sem mascara").
- ``upper``      — metade inferior apagada (caminho "mascara cirurgica").
- ``periocular`` — apenas a faixa dos olhos (caminho "balaclava/touca").

Todas as funcoes operam sobre o recorte ja alinhado em 112x112 pixels.
"""

from __future__ import annotations

import numpy as np


# Lado, em pixels, do recorte facial alinhado. O ArcFace espera 112x112.
ALIGNED_SIZE = 112

# Linhas de referencia no recorte alinhado, expressas como fracao da altura
# (0.0 = topo, 1.0 = base). Derivadas do template de 5 pontos do ArcFace:
# olhos em y~0.46, ponte do nariz em y~0.52.
EYES_Y = 0.46
MASK_TOP_Y = 0.52          # borda superior de uma mascara cirurgica
PERIOCULAR_TOP_Y = 0.34    # topo da faixa dos olhos
PERIOCULAR_BOTTOM_Y = 0.56 # base da faixa dos olhos

# Cor cinza neutra usada para apagar as regioes cobertas. A mesma cor e
# aplicada na galeria e no probe, mantendo os dois no mesmo dominio visual.
NEUTRAL_FILL = 128


def region_full(aligned_112: np.ndarray) -> np.ndarray:
    """Devolve o rosto completo, sem neutralizacao (caminho "sem mascara")."""

    return aligned_112.copy()


def region_upper(aligned_112: np.ndarray) -> np.ndarray:
    """Neutraliza a metade inferior do rosto (caminho "mascara parcial").

    Simula uma mascara cirurgica: nariz, boca e queixo sao apagados; olhos,
    sobrancelhas e testa permanecem visiveis.
    """

    out = aligned_112.copy()
    top = int(MASK_TOP_Y * ALIGNED_SIZE)
    out[top:, :] = NEUTRAL_FILL
    return out


def region_periocular(aligned_112: np.ndarray) -> np.ndarray:
    """Mantem apenas a faixa dos olhos (caminho "mascara ocular").

    Simula uma balaclava ou touca ninja: so a regiao em torno dos olhos
    permanece visivel; a testa e a metade inferior sao apagadas.
    """

    out = aligned_112.copy()
    top = int(PERIOCULAR_TOP_Y * ALIGNED_SIZE)
    bottom = int(PERIOCULAR_BOTTOM_Y * ALIGNED_SIZE)
    out[:top, :] = NEUTRAL_FILL
    out[bottom:, :] = NEUTRAL_FILL
    return out


# Mapa nome-da-variante -> funcao extratora. Quem precisa gerar as tres
# variantes itera sobre este dicionario, garantindo a mesma ordem em todo
# o codigo (galeria, webcam e avaliacao).
REGION_EXTRACTORS = {
    "full": region_full,
    "upper": region_upper,
    "periocular": region_periocular,
}
