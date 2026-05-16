"""Rastreador de rostos leve, por IoU e centroide.

Liga a caixa de um rosto entre frames a um ``track_id`` estavel. Sem isso nao
haveria onde acumular evidencia: cada frame seria uma analise isolada. O
rastreador e proposital simples e nao depende de bibliotecas externas de
tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Caixa delimitadora: cantos superior-esquerdo e inferior-direito.
Bbox = tuple[int, int, int, int]  # x1, y1, x2, y2


def iou(a: Bbox, b: Bbox) -> float:
    """Intersecao sobre uniao (IoU) entre duas caixas.

    Vale 1.0 quando as caixas coincidem e 0.0 quando nao se tocam. E a medida
    principal para decidir se duas deteccoes em frames diferentes sao o mesmo
    rosto.
    """

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def centroid(box: Bbox) -> tuple[float, float]:
    """Ponto central de uma caixa."""

    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def centroid_distance(a: Bbox, b: Bbox) -> float:
    """Distancia euclidiana entre os centros de duas caixas.

    Usada como desempate quando o IoU e zero — por exemplo, quando o rosto se
    moveu rapido o bastante para as caixas nao se sobreporem entre frames.
    """

    ax, ay = centroid(a)
    bx, by = centroid(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


@dataclass
class Track:
    """Um rosto acompanhado ao longo dos frames."""

    track_id: int          # identificador estavel deste rosto
    bbox: Bbox             # ultima caixa conhecida
    last_seen_frame: int   # indice do ultimo frame em que o rosto apareceu
    hits: int = 1          # quantas vezes o rosto foi associado a uma deteccao
    misses: int = 0        # frames seguidos sem deteccao correspondente

    @property
    def centroid(self) -> tuple[float, float]:
        return centroid(self.bbox)


@dataclass
class FaceTracker:
    """Associa deteccoes a tracks por IoU, com desempate por centroide."""

    iou_threshold: float = 0.3            # IoU minimo para considerar o mesmo rosto
    max_misses: int = 15                  # frames sem deteccao ate descartar o track
    max_centroid_distance: float = 80.0   # distancia maxima do desempate por centroide
    _tracks: dict[int, Track] = field(default_factory=dict)
    _next_id: int = 0

    def _new_track(self, bbox: Bbox, frame_index: int) -> int:
        """Cria um track novo para uma deteccao sem correspondencia."""

        track_id = self._next_id
        self._next_id += 1
        self._tracks[track_id] = Track(track_id, bbox, frame_index)
        return track_id

    def update(self, detections: list, frame_index: int) -> list[tuple[int, int]]:
        """Associa as deteccoes do frame atual aos tracks existentes.

        ``detections`` e uma lista de objetos com atributo ``.bbox``. Devolve
        uma lista de pares ``(track_id, indice_da_deteccao)``.

        A associacao acontece em duas passadas:
        1. Matching guloso por maior IoU.
        2. Para o que sobrou (IoU zero), desempate por menor distancia de
           centroide — cobre rostos que se moveram rapido.

        Deteccoes sem track viram tracks novos; tracks sem deteccao acumulam
        ``misses`` e sao removidos quando passam de ``max_misses``.
        """

        boxes = [tuple(int(v) for v in det.bbox) for det in detections]
        unmatched_dets = set(range(len(boxes)))
        unmatched_tracks = set(self._tracks)
        assignments: list[tuple[int, int]] = []

        # Passo 1: pares ordenados por maior IoU (matching guloso).
        pairs = []
        for det_index in unmatched_dets:
            for track_id in unmatched_tracks:
                score = iou(boxes[det_index], self._tracks[track_id].bbox)
                if score >= self.iou_threshold:
                    pairs.append((score, det_index, track_id))
        pairs.sort(reverse=True)
        for _, det_index, track_id in pairs:
            if det_index in unmatched_dets and track_id in unmatched_tracks:
                assignments.append((track_id, det_index))
                unmatched_dets.discard(det_index)
                unmatched_tracks.discard(track_id)

        # Passo 2: desempate por centroide para o que sobrou (IoU zero).
        leftovers = []
        for det_index in unmatched_dets:
            for track_id in unmatched_tracks:
                dist = centroid_distance(boxes[det_index], self._tracks[track_id].bbox)
                if dist <= self.max_centroid_distance:
                    leftovers.append((dist, det_index, track_id))
        leftovers.sort()
        for _, det_index, track_id in leftovers:
            if det_index in unmatched_dets and track_id in unmatched_tracks:
                assignments.append((track_id, det_index))
                unmatched_dets.discard(det_index)
                unmatched_tracks.discard(track_id)

        # Atualiza os tracks que foram associados a uma deteccao.
        for track_id, det_index in assignments:
            track = self._tracks[track_id]
            track.bbox = boxes[det_index]
            track.last_seen_frame = frame_index
            track.hits += 1
            track.misses = 0

        # Deteccoes sem track correspondente viram tracks novos.
        for det_index in unmatched_dets:
            track_id = self._new_track(boxes[det_index], frame_index)
            assignments.append((track_id, det_index))

        # Tracks sem deteccao acumulam misses e sao removidos se sumirem
        # por tempo demais.
        for track_id in unmatched_tracks:
            track = self._tracks[track_id]
            track.misses += 1
            if track.misses > self.max_misses:
                del self._tracks[track_id]

        return assignments

    def active_track_ids(self) -> set[int]:
        """Conjunto dos track_ids atualmente ativos."""

        return set(self._tracks)
