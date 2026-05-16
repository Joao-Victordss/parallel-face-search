"""Busca dos melhores candidatos: comparacao sequencial e paralela.

Dado o vetor facial de um rosto da webcam, este modulo o compara contra toda
a galeria e devolve os ``top_k`` candidatos mais parecidos. A mesma busca tem
duas implementacoes:

- Sequencial: um unico processo varre a galeria inteira.
- Paralela: a galeria e dividida entre varios processos, cada um varrendo a
  sua fatia; os resultados parciais sao combinados no fim.

Comparar as duas e a demonstracao de paralelismo do projeto. O parametro
``repeat`` repete a comparacao para aumentar a carga e tornar o ganho de
velocidade visivel no benchmark.
"""

from __future__ import annotations

import heapq
from concurrent.futures import ProcessPoolExecutor

from face_search.matching.candidate import Candidate, Match
from face_search.matching.similarity import cosine_similarity, similarity_to_score


# Galeria fatiada que cada processo de trabalho recebe na inicializacao.
# E uma variavel global de cada processo filho: definida uma vez por
# ``init_worker`` e reutilizada em todas as chamadas a ``compare_worker``,
# evitando reenviar a galeria a cada comparacao.
_WORKER_CANDIDATES: list[Candidate] = []


def top_matches_sequential(
    query_vector: tuple[float, ...],
    candidates: list[Candidate],
    variant: str,
    top_k: int,
    threshold: float,
    repeat: int,
) -> list[Match]:
    """Compara o vetor de consulta contra a galeria num unico processo.

    Mantem os ``top_k`` melhores num heap minimo: assim basta comparar a
    similaridade atual com a menor do heap, sem ordenar a galeria inteira.
    ``repeat`` repete a varredura para inflar a carga do benchmark.
    """

    best: list[tuple[float, int, Candidate]] = []

    for _ in range(repeat):
        best.clear()
        for index, candidate in enumerate(candidates):
            value = cosine_similarity(query_vector, candidate.vector_for(variant))
            item = (value, index, candidate)
            if len(best) < top_k:
                heapq.heappush(best, item)
            elif value > best[0][0]:
                heapq.heapreplace(best, item)

    return [
        Match(
            record_id=candidate.record_id,
            name=candidate.name,
            state=candidate.state,
            source_url=candidate.source_url,
            similarity=similarity,
            score=similarity_to_score(similarity, threshold),
        )
        for similarity, _, candidate in sorted(best, reverse=True)
    ]


def init_worker(candidates: list[Candidate]) -> None:
    """Inicializador de cada processo do pool: guarda a fatia da galeria."""

    global _WORKER_CANDIDATES
    _WORKER_CANDIDATES = candidates


def compare_worker(
    query_vector: tuple[float, ...],
    variant: str,
    top_k: int,
    threshold: float,
    repeat: int,
) -> list[Match]:
    """Funcao executada em cada processo: compara contra a sua fatia da galeria."""

    return top_matches_sequential(
        query_vector=query_vector,
        candidates=_WORKER_CANDIDATES,
        variant=variant,
        top_k=top_k,
        threshold=threshold,
        repeat=repeat,
    )


def split_candidates(
    candidates: list[Candidate],
    workers: int,
) -> list[list[Candidate]]:
    """Divide a galeria em ``workers`` fatias aproximadamente iguais.

    A distribuicao e intercalada (round-robin): cada fatia recebe um a cada
    ``workers`` candidatos, o que equilibra a carga entre os processos.
    """

    workers = max(1, min(workers, len(candidates)))
    chunks: list[list[Candidate]] = [[] for _ in range(workers)]
    for index, candidate in enumerate(candidates):
        chunks[index % workers].append(candidate)
    return chunks


def top_matches_parallel(
    query_vector: tuple[float, ...],
    executors: list[ProcessPoolExecutor],
    variant: str,
    top_k: int,
    threshold: float,
    repeat: int,
) -> list[Match]:
    """Compara o vetor de consulta contra a galeria usando varios processos.

    Cada executor compara contra a sua fatia em paralelo. Os ``top_k`` de cada
    fatia sao reunidos, reordenados por similaridade e cortados nos ``top_k``
    globais.
    """

    futures = [
        executor.submit(
            compare_worker, query_vector, variant, top_k, threshold, repeat
        )
        for executor in executors
    ]
    matches = [match for future in futures for match in future.result()]
    matches.sort(key=lambda item: item.similarity, reverse=True)
    return matches[:top_k]
