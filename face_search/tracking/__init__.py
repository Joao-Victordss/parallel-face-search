"""Rastreamento de rostos e acumulo de evidencia entre frames.

Este subpacote responde por dar continuidade temporal ao reconhecimento:

- ``tracker``  — ``FaceTracker``, que liga um rosto a um ``track_id`` estavel.
- ``evidence`` — ``EvidenceAccumulator``, que soma a evidencia de cada frame
  e transforma observacoes ruidosas numa confianca que cresce de forma suave.
"""

from __future__ import annotations

from face_search.tracking.evidence import (
    MASK_WEIGHTS,
    EvidenceAccumulator,
    clamp,
    frame_weight,
)
from face_search.tracking.tracker import (
    Bbox,
    FaceTracker,
    Track,
    centroid,
    centroid_distance,
    iou,
)

__all__ = [
    "MASK_WEIGHTS",
    "EvidenceAccumulator",
    "clamp",
    "frame_weight",
    "Bbox",
    "FaceTracker",
    "Track",
    "centroid",
    "centroid_distance",
    "iou",
]
