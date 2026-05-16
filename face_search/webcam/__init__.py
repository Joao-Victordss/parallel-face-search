"""Pipeline de busca facial pela webcam.

Este subpacote roda o reconhecimento em tempo real:

- ``camera``   — descoberta e abertura da webcam.
- ``hud``      — interface sobreposta ao video (mira, popups, chrome).
- ``pipeline`` — laco principal que junta deteccao, rastreamento, comparacao
  e acumulo de evidencia.
"""

from __future__ import annotations

from face_search.webcam.camera import (
    list_available_cameras,
    open_camera,
    probe_camera,
)
from face_search.webcam.pipeline import run_webcam

__all__ = [
    "list_available_cameras",
    "open_camera",
    "probe_camera",
    "run_webcam",
]
