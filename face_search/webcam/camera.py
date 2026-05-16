"""Descoberta e abertura da webcam.

O indice de cada camera varia por sistema e por quais dispositivos estao
conectados. Estas funcoes testam indices, listam as cameras disponiveis e
abrem a primeira que funcionar.
"""

from __future__ import annotations

import cv2


def probe_camera(index: int) -> bool:
    """Testa se existe uma camera utilizavel no indice informado."""

    capture = cv2.VideoCapture(index)
    try:
        return capture.isOpened()
    finally:
        capture.release()


def list_available_cameras(max_index: int) -> list[int]:
    """Devolve os indices de camera que respondem, de 0 ate ``max_index``."""

    return [index for index in range(max_index + 1) if probe_camera(index)]


def open_camera(camera: int, max_index: int) -> tuple[cv2.VideoCapture, int]:
    """Abre uma camera e devolve a captura junto com o indice usado.

    Com ``camera < 0``, varre os indices de 0 a ``max_index`` e abre o
    primeiro que funcionar. Com um indice especifico, tenta apenas aquele.
    Levanta ``RuntimeError`` com uma dica de diagnostico se nenhuma abrir.
    """

    indexes = range(max_index + 1) if camera < 0 else [camera]
    for index in indexes:
        capture = cv2.VideoCapture(index)
        if capture.isOpened():
            return capture, index
        capture.release()

    tried = f"0..{max_index}" if camera < 0 else str(camera)
    raise RuntimeError(
        f"nao foi possivel abrir camera nos indices {tried}. "
        "Verifique se existe /dev/video*, se outro app esta usando a camera "
        "ou rode com --list-cameras."
    )
