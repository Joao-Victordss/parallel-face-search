from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import face_recognition
import numpy as np
import psutil

from sync_mj_procurados import Config, object_key, r2_client


_WORKER_CANDIDATES: list["Candidate"] = []


@dataclass(frozen=True)
class Candidate:
    record_id: str
    name: str
    state: str
    source_url: str
    vector: tuple[float, ...]


@dataclass(frozen=True)
class Match:
    record_id: str
    name: str
    state: str
    source_url: str
    distance: float
    score: float


@dataclass
class RuntimeStats:
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
        if self.parallel_time <= 0:
            return 0.0
        return self.sequential_time / self.parallel_time


def distance_to_score(distance: float, threshold: float) -> float:
    if threshold <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - distance / threshold))


def distance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(a, b)))


def top_matches_sequential(
    query_vector: tuple[float, ...],
    candidates: list[Candidate],
    top_k: int,
    threshold: float,
    repeat: int,
) -> list[Match]:
    best: list[tuple[float, int, Candidate]] = []

    for _ in range(repeat):
        best.clear()
        for index, candidate in enumerate(candidates):
            value = distance(query_vector, candidate.vector)
            item = (-value, index, candidate)
            if len(best) < top_k:
                heapq.heappush(best, item)
            elif item > best[0]:
                heapq.heapreplace(best, item)

    return [
        Match(
            record_id=candidate.record_id,
            name=candidate.name,
            state=candidate.state,
            source_url=candidate.source_url,
            distance=-negative_distance,
            score=distance_to_score(-negative_distance, threshold),
        )
        for negative_distance, _, candidate in sorted(best, reverse=True)
    ]


def init_worker(candidates: list[Candidate]) -> None:
    global _WORKER_CANDIDATES
    _WORKER_CANDIDATES = candidates


def compare_worker(
    query_vector: tuple[float, ...],
    top_k: int,
    threshold: float,
    repeat: int,
) -> list[Match]:
    return top_matches_sequential(
        query_vector=query_vector,
        candidates=_WORKER_CANDIDATES,
        top_k=top_k,
        threshold=threshold,
        repeat=repeat,
    )


def split_candidates(candidates: list[Candidate], workers: int) -> list[list[Candidate]]:
    workers = max(1, min(workers, len(candidates)))
    chunks = [[] for _ in range(workers)]
    for index, candidate in enumerate(candidates):
        chunks[index % workers].append(candidate)
    return chunks


def top_matches_parallel(
    query_vector: tuple[float, ...],
    executors: list[ProcessPoolExecutor],
    top_k: int,
    threshold: float,
    repeat: int,
) -> list[Match]:
    futures = [
        executor.submit(compare_worker, query_vector, top_k, threshold, repeat)
        for executor in executors
    ]
    matches = [match for future in futures for match in future.result()]
    matches.sort(key=lambda item: item.distance)
    return matches[:top_k]


def probe_camera(index: int) -> bool:
    capture = cv2.VideoCapture(index)
    try:
        return capture.isOpened()
    finally:
        capture.release()


def list_available_cameras(max_index: int) -> list[int]:
    return [index for index in range(max_index + 1) if probe_camera(index)]


def open_camera(camera: int, max_index: int) -> tuple[cv2.VideoCapture, int]:
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


def load_manifest_from_path(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest_from_r2() -> dict[str, Any]:
    config = Config.from_env(limit=None, no_upload=False)
    client = r2_client(config)
    response = client.get_object(
        Bucket=config.r2_bucket,
        Key=object_key(config, "manifest.json"),
    )
    return json.loads(response["Body"].read().decode("utf-8"))


def load_candidates(manifest: dict[str, Any]) -> list[Candidate]:
    candidates = []
    for record in manifest.get("records", []):
        vector = record.get("face_vector")
        if not isinstance(vector, list) or len(vector) != 128:
            continue
        candidates.append(
            Candidate(
                record_id=record["id"],
                name=record["name"],
                state=record.get("state") or "",
                source_url=record["source_url"],
                vector=tuple(float(value) for value in vector),
            )
        )
    if not candidates:
        raise RuntimeError("nenhum vetor facial valido encontrado no manifesto")
    return candidates


def encode_frame_faces(
    frame_bgr: np.ndarray,
    frame_scale: float,
    detection_model: str,
) -> tuple[list[tuple[int, int, int, int]], list[tuple[float, ...]]]:
    small = cv2.resize(frame_bgr, (0, 0), fx=frame_scale, fy=frame_scale)
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    small_locations = face_recognition.face_locations(rgb_small, model=detection_model)
    encodings = face_recognition.face_encodings(
        rgb_small,
        known_face_locations=small_locations,
        num_jitters=1,
        model="small",
    )

    scale = 1 / frame_scale
    locations = [
        (
            int(top * scale),
            int(right * scale),
            int(bottom * scale),
            int(left * scale),
        )
        for top, right, bottom, left in small_locations
    ]
    vectors = [tuple(float(value) for value in encoding) for encoding in encodings]
    return locations, vectors


def draw_match(
    frame: np.ndarray,
    location: tuple[int, int, int, int],
    match: Match | None,
    threshold: float,
) -> None:
    top, right, bottom, left = location
    is_match = match is not None and match.distance <= threshold
    color = (0, 0, 255) if is_match else (0, 180, 255)
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

    if match is None:
        label = "sem match"
    else:
        label = f"{match.name} ({match.state}) {match.score * 100:.1f}% d={match.distance:.3f}"

    y = max(24, top - 8)
    cv2.putText(
        frame,
        label,
        (left, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_status(
    frame: np.ndarray,
    mode: str,
    stats: RuntimeStats,
    started_at: float,
    threshold: float,
    workers: int,
) -> None:
    elapsed = max(0.001, time.perf_counter() - started_at)
    fps = stats.frames / elapsed
    cpu = psutil.cpu_percent(interval=None)
    text = (
        f"modo={mode} fps={fps:.1f} cpu={cpu:.0f}% threshold={threshold:.2f} "
        f"seq={stats.avg_sequential_ms():.1f}ms par={stats.avg_parallel_ms():.1f}ms "
        f"speedup={stats.speedup():.2f}x workers={workers}"
    )
    cv2.putText(
        frame,
        text,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def compare_face(
    query_vector: tuple[float, ...],
    candidates: list[Candidate],
    executors: list[ProcessPoolExecutor],
    mode: str,
    top_k: int,
    threshold: float,
    repeat: int,
    stats: RuntimeStats,
) -> list[Match]:
    if mode == "sequential":
        start = time.perf_counter()
        matches = top_matches_sequential(query_vector, candidates, top_k, threshold, repeat)
        stats.sequential_time += time.perf_counter() - start
        return matches

    if mode == "parallel":
        if not executors:
            raise RuntimeError("executor paralelo nao inicializado")
        start = time.perf_counter()
        matches = top_matches_parallel(
            query_vector,
            executors,
            top_k,
            threshold,
            repeat,
        )
        stats.parallel_time += time.perf_counter() - start
        return matches

    start = time.perf_counter()
    sequential_matches = top_matches_sequential(
        query_vector,
        candidates,
        top_k,
        threshold,
        repeat,
    )
    stats.sequential_time += time.perf_counter() - start

    if not executors:
        raise RuntimeError("executor paralelo nao inicializado")
    start = time.perf_counter()
    parallel_matches = top_matches_parallel(
        query_vector,
        executors,
        top_k,
        threshold,
        repeat,
    )
    stats.parallel_time += time.perf_counter() - start
    return parallel_matches or sequential_matches


def run_webcam(args: argparse.Namespace) -> None:
    validate_args(args)
    capture, camera_index = open_camera(args.camera, args.max_camera_index)

    manifest = (
        load_manifest_from_path(args.manifest)
        if args.manifest
        else load_manifest_from_r2()
    )
    candidates = load_candidates(manifest)
    worker_count = max(1, args.workers or (os.cpu_count() or 1))
    worker_chunks = split_candidates(candidates, worker_count)

    executors: list[ProcessPoolExecutor] = []
    if args.mode in {"parallel", "benchmark"}:
        executors = [
            ProcessPoolExecutor(
                max_workers=1,
                initializer=init_worker,
                initargs=(chunk,),
            )
            for chunk in worker_chunks
        ]

    print(f"base carregada: {len(candidates)} vetores")
    print(f"camera aberta: {camera_index}")
    print("pressione q para sair")

    stats = RuntimeStats()
    started_at = time.perf_counter()
    last_locations: list[tuple[int, int, int, int]] = []
    last_matches: list[Match | None] = []

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            should_process = stats.frames % args.process_every == 0
            if should_process:
                locations, vectors = encode_frame_faces(
                    frame,
                    frame_scale=args.frame_scale,
                    detection_model=args.detection_model,
                )
                matches: list[Match | None] = []
                for vector in vectors:
                    result = compare_face(
                        query_vector=vector,
                        candidates=candidates,
                        executors=executors,
                        mode=args.mode,
                        top_k=args.top_k,
                        threshold=args.threshold,
                        repeat=args.repeat,
                        stats=stats,
                    )
                    stats.compared_faces += 1
                    matches.append(result[0] if result else None)

                last_locations = locations
                last_matches = matches

            for location, match in zip(last_locations, last_matches):
                draw_match(frame, location, match, args.threshold)

            stats.frames += 1
            draw_status(
                frame,
                mode=args.mode,
                stats=stats,
                started_at=started_at,
                threshold=args.threshold,
                workers=len(worker_chunks),
            )
            cv2.imshow("Parallel Face Search", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compara rostos da webcam contra os vetores do manifesto."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Caminho para manifest.json local. Se omitido, carrega do R2.",
    )
    parser.add_argument(
        "--mode",
        choices=("sequential", "parallel", "benchmark"),
        default="parallel",
        help="Modo de comparacao.",
    )
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument(
        "--camera",
        type=int,
        default=-1,
        help="Indice da camera. Use -1 para tentar automaticamente.",
    )
    parser.add_argument(
        "--max-camera-index",
        type=int,
        default=5,
        help="Maior indice testado no modo automatico e no --list-cameras.",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="Lista cameras disponiveis e encerra.",
    )
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--process-every", type=int, default=5)
    parser.add_argument("--frame-scale", type=float, default=0.25)
    parser.add_argument("--detection-model", choices=("hog", "cnn"), default="hog")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.frame_scale <= 0 or args.frame_scale > 1:
        raise ValueError("--frame-scale deve ficar entre 0 e 1")
    if args.process_every < 1:
        raise ValueError("--process-every deve ser maior ou igual a 1")
    if args.repeat < 1:
        raise ValueError("--repeat deve ser maior ou igual a 1")
    if args.top_k < 1:
        raise ValueError("--top-k deve ser maior ou igual a 1")
    if args.workers < 1:
        raise ValueError("--workers deve ser maior ou igual a 1")
    if args.max_camera_index < 0:
        raise ValueError("--max-camera-index deve ser maior ou igual a 0")


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.list_cameras:
        cameras = list_available_cameras(parsed_args.max_camera_index)
        if cameras:
            print("cameras disponiveis:", ", ".join(str(index) for index in cameras))
        else:
            print("nenhuma camera encontrada")
        raise SystemExit(0)

    run_webcam(parsed_args)
