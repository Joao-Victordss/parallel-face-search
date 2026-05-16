"""Comparacao de rostos contra a galeria.

Este subpacote responde pela pergunta "este rosto e algum procurado?":

- ``similarity`` — similaridade de cosseno (em Python puro, de proposito).
- ``candidate``  — ``Candidate``/``Match`` e o carregamento do manifesto.
- ``search``     — busca dos melhores candidatos, sequencial e paralela.
"""

from __future__ import annotations

from face_search.matching.candidate import (
    MATCH_METHODS,
    VARIANTS,
    Candidate,
    Match,
    load_candidates,
    load_manifest_from_path,
    load_manifest_from_r2,
)
from face_search.matching.search import (
    compare_worker,
    init_worker,
    split_candidates,
    top_matches_parallel,
    top_matches_sequential,
)
from face_search.matching.similarity import cosine_similarity, similarity_to_score

__all__ = [
    "MATCH_METHODS",
    "VARIANTS",
    "Candidate",
    "Match",
    "load_candidates",
    "load_manifest_from_path",
    "load_manifest_from_r2",
    "compare_worker",
    "init_worker",
    "split_candidates",
    "top_matches_parallel",
    "top_matches_sequential",
    "cosine_similarity",
    "similarity_to_score",
]
