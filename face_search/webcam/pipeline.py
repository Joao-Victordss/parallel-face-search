"""Pipeline de busca facial pela webcam.

Este modulo junta todas as pecas do projeto num laco de tempo real. Para cada
frame da webcam:

1. O SCRFD detecta e alinha cada rosto.
2. O tracker associa cada rosto a um ``track_id`` estavel.
3. As tres regioes do rosto sao codificadas com o ArcFace numa unica inferencia.
4. Os tres caminhos de comparacao rodam (sequencial, paralelo ou benchmark).
5. O acumulador soma a evidencia ao track; a confianca cresce com frames
   consistentes.
6. O HUD desenha o nome do candidato, a confianca e o estado.

Para nao travar o video quando a inferencia demora, a captura/exibicao roda
na thread principal e o reconhecimento roda numa thread separada.

O resultado e sempre um candidato para verificacao humana, nunca uma
confirmacao automatica de identidade.
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import cv2

from face_search import engine
from face_search.engine import DetectedFace, EngineConfig
from face_search.matching import (
    MATCH_METHODS,
    Candidate,
    Match,
    init_worker,
    load_candidates,
    load_manifest_from_path,
    load_manifest_from_r2,
    split_candidates,
    top_matches_parallel,
    top_matches_sequential,
)
from face_search.tracking import EvidenceAccumulator, FaceTracker, frame_weight
from face_search.webcam.camera import open_camera
from face_search.webcam.hud import (
    STATUS_CONFIRMADO,
    STATUS_OK,
    STATUS_PROVAVEL,
    TrackView,
    draw_hud,
    draw_match,
)


@dataclass
class FrameFace:
    """Um rosto detectado num frame, ja com o recorte alinhado 112x112."""

    face: DetectedFace
    aligned: Any  # recorte BGR 112x112


@dataclass
class LostTrack:
    """Um track que sumiu da cena, guardado para re-identificacao.

    Se o mesmo rosto voltar, ele herda este acumulador e a confianca continua
    de onde parou, em vez de zerar.
    """

    embedding: Any                       # embedding do rosto exposto (identidade)
    accumulator: EvidenceAccumulator
    lost_frame: int                      # frame em que o track sumiu


@dataclass
class RuntimeStats:
    """Estatisticas de execucao, usadas no benchmark e no resumo final."""

    frames: int = 0
    compared_faces: int = 0
    sequential_time: float = 0.0
    parallel_time: float = 0.0

    def avg_sequential_ms(self) -> float:
        if self.compared_faces == 0:
            return 0.0
        return 1000 * self.sequential_time / self.compared_faces

    def avg_parallel_ms(self) -> float:
        if self.compared_faces == 0:
            return 0.0
        return 1000 * self.parallel_time / self.compared_faces

    def speedup(self) -> float:
        """Ganho de velocidade do modo paralelo sobre o sequencial."""

        if self.parallel_time <= 0:
            return 0.0
        return self.sequential_time / self.parallel_time


def detect_and_align(
    frame_bgr: Any,
    frame_scale: float,
    engine_config: EngineConfig,
) -> list[FrameFace]:
    """Detecta os rostos de um frame e devolve cada um com o recorte alinhado.

    A deteccao roda numa versao reduzida do frame (``frame_scale``), o que
    acelera o SCRFD. As coordenadas sao reescaladas de volta para o frame
    original, mas o recorte alinhado e feito sobre o frame em resolucao cheia,
    preservando a qualidade do embedding.
    """

    small = cv2.resize(frame_bgr, (0, 0), fx=frame_scale, fy=frame_scale)
    detections = engine.detect_faces(small, engine_config)

    scale = 1.0 / frame_scale
    frame_faces: list[FrameFace] = []
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        bbox = (int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale))
        kps = detection.kps * scale
        face = DetectedFace(bbox=bbox, kps=kps, det_score=detection.det_score)
        aligned = engine.align_crop(frame_bgr, kps)
        frame_faces.append(FrameFace(face=face, aligned=aligned))
    return frame_faces


def compare_face(
    query_vector: tuple[float, ...],
    candidates: list[Candidate],
    executors: list[ProcessPoolExecutor],
    mode: str,
    variant: str,
    top_k: int,
    threshold: float,
    repeat: int,
    stats: RuntimeStats,
) -> list[Match]:
    """Compara um vetor contra a galeria, conforme o modo escolhido.

    - ``sequential``: roda apenas a busca sequencial e cronometra.
    - ``parallel``: roda apenas a busca paralela e cronometra.
    - ``benchmark``: roda as duas, cronometra ambas e devolve o resultado
      paralelo (ou o sequencial, se o paralelo vier vazio).

    Os tempos sao somados em ``stats`` para o relatorio de speedup.
    """

    if mode == "sequential":
        start = time.perf_counter()
        matches = top_matches_sequential(
            query_vector, candidates, variant, top_k, threshold, repeat
        )
        stats.sequential_time += time.perf_counter() - start
        return matches

    if mode == "parallel":
        if not executors:
            raise RuntimeError("executor paralelo nao inicializado")
        start = time.perf_counter()
        matches = top_matches_parallel(
            query_vector, executors, variant, top_k, threshold, repeat
        )
        stats.parallel_time += time.perf_counter() - start
        return matches

    # Modo benchmark: roda os dois e mede cada um.
    start = time.perf_counter()
    sequential_matches = top_matches_sequential(
        query_vector, candidates, variant, top_k, threshold, repeat
    )
    stats.sequential_time += time.perf_counter() - start

    if not executors:
        raise RuntimeError("executor paralelo nao inicializado")
    start = time.perf_counter()
    parallel_matches = top_matches_parallel(
        query_vector, executors, variant, top_k, threshold, repeat
    )
    stats.parallel_time += time.perf_counter() - start
    return parallel_matches or sequential_matches


def build_view(
    face: DetectedFace,
    accumulator: EvidenceAccumulator,
    candidates_by_id: dict[str, Candidate],
    confirm_threshold: float,
    min_frames: int,
) -> TrackView:
    """Monta a ``TrackView`` de um track a partir do seu acumulador.

    O estado e decidido pela evidencia acumulada: sem candidato vira
    ``STATUS_OK``; com candidato mas ainda nao confirmado, ``STATUS_PROVAVEL``;
    confirmado, ``STATUS_CONFIRMADO``.
    """

    leading_id, _ = accumulator.leading()
    candidate = candidates_by_id.get(leading_id) if leading_id else None
    confidence = accumulator.confidence()

    if candidate is None or confidence <= 0.0:
        status = STATUS_OK
    elif accumulator.is_confident(confirm_threshold, min_frames):
        status = STATUS_CONFIRMADO
    else:
        status = STATUS_PROVAVEL

    return TrackView(
        bbox=face.bbox,
        candidate_name=candidate.name if candidate else None,
        confidence=confidence,
        status=status,
    )


def reidentify(
    embedding: Any,
    lost_tracks: list[LostTrack],
    threshold: float,
) -> EvidenceAccumulator | None:
    """Procura, entre os tracks perdidos, um rosto igual ao do embedding dado.

    Se achar um com similaridade de cosseno acima do limiar, devolve o
    acumulador dele e o remove da lista, para o track novo continuar de onde
    o antigo parou. Como os embeddings ja vem normalizados, o produto interno
    e a propria similaridade de cosseno.
    """

    import numpy as np

    best: LostTrack | None = None
    best_similarity = threshold
    for lost in lost_tracks:
        similarity = float(np.dot(embedding, lost.embedding))
        if similarity >= best_similarity:
            best_similarity = similarity
            best = lost
    if best is None:
        return None
    lost_tracks.remove(best)
    return best.accumulator


def run_webcam(args: argparse.Namespace) -> None:
    """Executa o laco principal de busca facial pela webcam.

    Recebe os parametros ja parseados e validados pela camada de CLI.
    """

    engine_config = EngineConfig(
        onnx_provider=args.onnx_provider,
        det_size=args.det_size,
        min_face=args.min_face,
    )
    capture, camera_index = open_camera(args.camera, args.max_camera_index)
    # Buffer de 1 frame: a webcam descarta frames antigos e sempre entrega o
    # mais recente, evitando que o video atrase em relacao a realidade.
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # --- Galeria -----------------------------------------------------------
    manifest = (
        load_manifest_from_path(args.manifest)
        if args.manifest
        else load_manifest_from_r2()
    )
    candidates = load_candidates(manifest)
    candidates_by_id = {candidate.record_id: candidate for candidate in candidates}

    # --- Pool de processos para a comparacao paralela ----------------------
    worker_count = max(1, args.workers or (os.cpu_count() or 1))
    worker_chunks = split_candidates(candidates, worker_count)
    executors: list[ProcessPoolExecutor] = []
    if args.mode in {"parallel", "benchmark"}:
        executors = [
            ProcessPoolExecutor(
                max_workers=1, initializer=init_worker, initargs=(chunk,)
            )
            for chunk in worker_chunks
        ]

    # Um threshold de comparacao por caminho.
    thresholds = {
        "full": args.threshold_full,
        "upper": args.threshold_upper,
        "periocular": args.threshold_periocular,
    }

    # --- Estruturas de rastreamento e evidencia ----------------------------
    tracker = FaceTracker(iou_threshold=args.iou_threshold, max_misses=args.max_misses)
    # Um acumulador por track: os tres caminhos somam evidencia ao mesmo.
    accumulators: dict[int, EvidenceAccumulator] = {}
    # Embedding atual de cada track e a lista de tracks perdidos, usados para
    # re-identificar um rosto que sai e volta sem zerar a confianca.
    track_embeddings: dict[int, Any] = {}
    lost_tracks: list[LostTrack] = []
    stats = RuntimeStats()

    # --- Comunicacao entre as threads --------------------------------------
    # A captura/exibicao roda na thread principal; o reconhecimento roda na
    # thread abaixo. A thread de reconhecimento sempre pega o frame mais
    # recente e publica os overlays quando termina; a tela segue fluida.
    state_lock = threading.Lock()
    shared: dict[str, Any] = {"frame": None, "index": 0, "views": []}
    stop_event = threading.Event()

    def recognition_loop() -> None:
        """Laco da thread de reconhecimento (detecta, codifica e compara)."""

        processed_index = -1
        worker_frames = 0
        while not stop_event.is_set():
            with state_lock:
                frame = shared["frame"]
                index = shared["index"]
            # Espera ate haver um frame novo para processar.
            if frame is None or index == processed_index:
                time.sleep(0.004)
                continue
            processed_index = index
            worker_frames += 1

            frame_faces = detect_and_align(frame, args.frame_scale, engine_config)
            assignments = tracker.update(
                [ff.face for ff in frame_faces], worker_frames
            )

            views: list[TrackView] = []
            for track_id, det_index in assignments:
                frame_face = frame_faces[det_index]
                face = frame_face.face
                aligned = frame_face.aligned
                quality = engine.quality_gate(aligned, face, engine_config)

                # Codifica as tres regioes numa unica inferencia ArcFace.
                regions = [
                    engine.REGION_EXTRACTORS[variant](aligned)
                    for _, variant in MATCH_METHODS
                ]
                vectors = engine.embed_batch(regions, engine_config)
                vector_by_variant = {
                    variant: vector
                    for (_, variant), vector in zip(MATCH_METHODS, vectors)
                }
                # O embedding do rosto exposto serve de identidade do track.
                identity_vector = vector_by_variant["full"]

                # Track novo tenta herdar o acumulador de um track perdido com
                # o mesmo rosto, para a confianca nao reiniciar quando alguem
                # sai e volta para a cena.
                if track_id not in accumulators:
                    accumulators[track_id] = reidentify(
                        identity_vector, lost_tracks, args.reid_threshold
                    ) or EvidenceAccumulator(
                        decay=args.decay,
                        confidence_k=args.confidence_k,
                    )
                accumulator = accumulators[track_id]
                track_embeddings[track_id] = identity_vector

                # Os tres caminhos de match comparam de forma separada; cada
                # um que encontra candidato contribui para a mesma confianca.
                contributions: list[tuple[str, float]] = []
                for _, variant in MATCH_METHODS:
                    threshold = thresholds[variant]
                    query_vector = tuple(
                        float(v) for v in vector_by_variant[variant]
                    )
                    matches = compare_face(
                        query_vector=query_vector,
                        candidates=candidates,
                        executors=executors,
                        mode=args.mode,
                        variant=variant,
                        top_k=args.top_k,
                        threshold=threshold,
                        repeat=args.repeat,
                        stats=stats,
                    )
                    stats.compared_faces += 1

                    # Teste de razao top-1/top-2: so aceita o melhor candidato
                    # se ele estiver suficientemente a frente do segundo.
                    top = matches[0] if matches else None
                    gap_ok = len(matches) < 2 or (
                        matches[0].similarity - matches[1].similarity
                        >= args.ratio_gap
                    )
                    if top is not None and gap_ok and top.similarity >= threshold:
                        weight = frame_weight(
                            variant, quality.focus, args.ref_focus, face.det_score
                        )
                        contributions.append(
                            (top.record_id, weight * (top.similarity - threshold))
                        )

                accumulator.observe(contributions, worker_frames)
                views.append(
                    build_view(
                        face=face,
                        accumulator=accumulator,
                        candidates_by_id=candidates_by_id,
                        confirm_threshold=args.confirm_threshold,
                        min_frames=args.min_frames,
                    )
                )

            # Tracks que sumiram vao para a lista de perdidos, com o seu
            # acumulador, para poderem ser re-identificados se voltarem.
            active_ids = tracker.active_track_ids()
            for track_id in list(accumulators):
                if track_id not in active_ids:
                    embedding = track_embeddings.pop(track_id, None)
                    if embedding is not None:
                        lost_tracks.append(
                            LostTrack(
                                embedding=embedding,
                                accumulator=accumulators[track_id],
                                lost_frame=worker_frames,
                            )
                        )
                    del accumulators[track_id]
            # Descarta tracks perdidos antigos demais para re-identificar.
            lost_tracks[:] = [
                lost
                for lost in lost_tracks
                if worker_frames - lost.lost_frame <= args.reid_max_age
            ]

            with state_lock:
                shared["views"] = views

    print(f"base carregada: {len(candidates)} procurados")
    print(f"camera aberta: {camera_index}")
    print("pressione q para sair")

    worker = threading.Thread(target=recognition_loop, daemon=True)
    worker.start()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            with state_lock:
                shared["frame"] = frame
                shared["index"] += 1
                views = shared["views"]

            # Desenha os overlays mais recentes sobre uma copia do frame.
            display = frame.copy()
            for view in views:
                draw_match(display, view)
            stats.frames += 1
            draw_hud(display, stats.frames)
            cv2.imshow("Parallel Face Search", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        stop_event.set()
        worker.join(timeout=3.0)
        capture.release()
        cv2.destroyAllWindows()
        for executor in executors:
            executor.shutdown()

    print(
        "resultado: "
        f"faces={stats.compared_faces}, "
        f"seq={stats.avg_sequential_ms():.2f}ms, "
        f"par={stats.avg_parallel_ms():.2f}ms, "
        f"speedup={stats.speedup():.2f}x"
    )
