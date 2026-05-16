"""Interface sobreposta ao video (HUD) da webcam.

Este modulo so desenha. Ele recebe o que ja foi decidido (a ``TrackView`` de
cada rosto) e cuida da parte visual: a mira sobre cada rosto, o popup com
nome e confianca, e o "chrome" do HUD (cantos da tela e linha de varredura).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


# --- Estados de um rosto na tela -------------------------------------------
# Verde: rosto detectado, sem match. Amarelo: candidato em formacao, ainda
# nao confirmado. Vermelho: match confirmado (sempre para verificacao humana).
STATUS_OK = "ok"
STATUS_PROVAVEL = "provavel"
STATUS_CONFIRMADO = "confirmado"

# Cor de cada estado, no formato BGR usado pelo OpenCV.
STATUS_COLOR = {
    STATUS_OK: (0, 200, 0),
    STATUS_PROVAVEL: (0, 215, 255),
    STATUS_CONFIRMADO: (0, 0, 255),
}

# --- Tipografia e cores do chrome ------------------------------------------
FONT = cv2.FONT_HERSHEY_DUPLEX
HUD_ACCENT = (210, 190, 90)   # ciano-aco frio, usado nas bordas do HUD
HUD_TEXT = (235, 235, 235)    # cinza claro, usado nos textos secundarios


@dataclass
class TrackView:
    """Tudo o que o HUD precisa para desenhar um rosto rastreado."""

    bbox: tuple[int, int, int, int]  # caixa do rosto no frame
    candidate_name: str | None       # nome do candidato lider (ou None)
    confidence: float                # confianca acumulada, em [0, 1]
    status: str                      # um dos STATUS_* acima


def _panel(frame: Any, x1: int, y1: int, x2: int, y2: int, alpha: float = 0.62) -> None:
    """Escurece um retangulo do frame, criando um painel translucido.

    Usado como fundo dos textos, para que eles fiquem legiveis sobre
    qualquer cena. ``alpha`` controla a intensidade do escurecimento.
    """

    height, width = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return
    region = frame[y1:y2, x1:x2]
    region[:] = (region.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)


def _corner_brackets(
    frame: Any,
    bbox: tuple[int, int, int, int],
    color,
    thickness: int = 2,
) -> None:
    """Desenha cantos em "L" em vez de um retangulo fechado (mira de alvo)."""

    left, top, right, bottom = bbox
    seg = max(14, int(0.22 * min(right - left, bottom - top)))
    for cx, cy, dx, dy in (
        (left, top, 1, 1),
        (right, top, -1, 1),
        (left, bottom, 1, -1),
        (right, bottom, -1, -1),
    ):
        cv2.line(frame, (cx, cy), (cx + dx * seg, cy), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy * seg), color, thickness, cv2.LINE_AA)


def _crosshair(frame: Any, center: tuple[int, int], color) -> None:
    """Desenha uma mira em cruz no centro do rosto."""

    cx, cy = center
    gap, arm = 5, 12
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        cv2.line(
            frame,
            (cx + dx * gap, cy + dy * gap),
            (cx + dx * arm, cy + dy * arm),
            color,
            1,
            cv2.LINE_AA,
        )


def draw_match(frame: Any, view: TrackView) -> None:
    """Desenha a mira de um rosto e o popup com nome e confianca.

    A confianca exibida e a soma da evidencia dos tres caminhos de comparacao.
    Um rosto que nao casa com nenhuma foto indexada aparece como
    "Desconhecido".
    """

    left, top, right, bottom = view.bbox
    color = STATUS_COLOR[view.status]
    _corner_brackets(frame, view.bbox, color)
    _crosshair(frame, ((left + right) // 2, (top + bottom) // 2), color)

    matched = view.status != STATUS_OK and view.candidate_name
    name = view.candidate_name if matched else "Desconhecido"
    confidence = f"Confianca {view.confidence * 100:.0f}%"

    # Dimensiona o painel pelo maior dos dois textos.
    name_scale, conf_scale = 0.62, 0.46
    name_width = cv2.getTextSize(name, FONT, name_scale, 1)[0][0]
    conf_width = cv2.getTextSize(confidence, FONT, conf_scale, 1)[0][0]
    block_w = max(name_width, conf_width) + 20
    block_h = 48

    bx1 = left
    by1 = max(0, top - block_h - 6)
    _panel(frame, bx1, by1, bx1 + block_w, by1 + block_h, alpha=0.58)
    cv2.line(frame, (bx1, by1), (bx1, by1 + block_h), color, 2, cv2.LINE_AA)
    cv2.putText(
        frame, name, (bx1 + 10, by1 + 23), FONT, name_scale, color, 1, cv2.LINE_AA
    )
    cv2.putText(
        frame, confidence, (bx1 + 10, by1 + 41), FONT, conf_scale, HUD_TEXT, 1,
        cv2.LINE_AA,
    )


def draw_hud(frame: Any, frame_index: int) -> None:
    """Desenha o chrome do HUD: os cantos da tela e a linha de varredura.

    A linha de varredura e puramente estetica: uma faixa horizontal que sobe
    e desce com o passar dos frames, dando ao video a aparencia de um sistema
    de monitoramento ativo.
    """

    height, width = frame.shape[:2]

    # Cantos em "L" nas quatro pontas da tela.
    tick = 26
    for cx, cy, dx, dy in (
        (1, 1, 1, 1),
        (width - 2, 1, -1, 1),
        (1, height - 2, 1, -1),
        (width - 2, height - 2, -1, -1),
    ):
        cv2.line(frame, (cx, cy), (cx + dx * tick, cy), HUD_ACCENT, 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy * tick), HUD_ACCENT, 1, cv2.LINE_AA)

    # Linha de varredura: clareia uma faixa fina que percorre a altura.
    period = 130
    scan_y = int((frame_index % period) / period * height)
    band = frame[max(0, scan_y - 1):min(height, scan_y + 2), :]
    if band.size:
        band[:] = np.clip(band.astype(np.int16) + 40, 0, 255).astype(np.uint8)
