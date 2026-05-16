"""Harness de avaliacao de acuracia do reconhecimento facial.

Mede a qualidade da comparacao para os tres caminhos do pipeline (sem
mascara, mascara parcial, mascara ocular) e sugere os thresholds.

Limitacao honesta: o manifesto traz apenas a foto oficial de cada procurado,
sem uma segunda imagem por pessoa. Sem isso nao e possivel medir a acuracia
real de ponta a ponta. Este harness faz uma avaliacao sintetica a partir das
variantes de embedding do proprio manifesto:

- probe   = embedding da variante mascarada (upper ou periocular) de A
- galeria = embedding completo (full) de todos os procurados

Como a variante mascarada passou por neutralizacao de regiao, ela difere do
embedding completo, entao o ranqueamento nao e trivial. O resultado e um
limite superior otimista da acuracia real: um probe de verdade vindo da
webcam e outra foto, mais ruidosa. Para metricas reais ponta a ponta, capture
um conjunto de probes rotulados e estenda este modulo.

Saida: relatorio JSON deterministico (seed fixa) e um resumo no terminal.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from face_search.engine import EMBEDDING_DIM, EMBEDDING_MODEL
from face_search.matching import Candidate, cosine_similarity


# Seed fixa: garante que o relatorio seja deterministico entre execucoes.
SEED = 20260516

# Cada cenario avalia um caminho do pipeline. O probe usa a variante indicada;
# a galeria usa sempre o embedding completo (foto oficial sem mascara).
SCENARIOS = {
    "sem_mascara": "full",
    "mascara_parcial": "upper",
    "mascara_ocular": "periocular",
}


@dataclass
class ScenarioReport:
    """Metricas calculadas para um cenario (um caminho do pipeline)."""

    scenario: str
    probe_variant: str
    degenerate: bool                 # True quando probe == galeria (genuino trivial)
    rank1_accuracy: float            # fracao de probes que ranqueiam a si mesmos em 1o
    genuine_count: int
    impostor_count: int
    genuine_mean: float              # similaridade media dos pares genuinos
    impostor_mean: float             # similaridade media dos pares impostores
    impostor_p99: float              # similaridade no percentil 99 dos impostores
    eer: float                       # equal error rate
    eer_threshold: float             # threshold no ponto de EER
    suggested_threshold: float       # threshold sugerido para uso pratico
    far_at_suggested: float          # falsa aceitacao no threshold sugerido
    frr_at_suggested: float          # falsa rejeicao no threshold sugerido
    precision_at_suggested: float
    recall_at_suggested: float


def collect_pairs(
    candidates: list[Candidate],
    probe_variant: str,
    rng: random.Random,
    max_impostors: int,
) -> tuple[list[float], list[float], float]:
    """Coleta similaridades genuinas e impostoras e calcula a acuracia rank-1.

    - Genuino: probe (variante) de A contra a galeria completa de A.
    - Impostor: probe (variante) de A contra a galeria completa de outros.
    - Rank-1: o probe de A ranqueia a galeria completa de A em primeiro lugar.

    Os impostores sao amostrados (ate ``max_impostors`` por identidade) para
    manter o relatorio rapido mesmo com galerias grandes.
    """

    genuine: list[float] = []
    impostor: list[float] = []
    rank1_hits = 0

    for index, person in enumerate(candidates):
        probe = person.vector_for(probe_variant)

        # Par genuino: o probe contra a propria galeria.
        genuine_similarity = cosine_similarity(probe, person.vector_full)
        genuine.append(genuine_similarity)

        # Rank-1: o probe precisa bater a galeria de todos os outros.
        best_other = -2.0
        others = [c for position, c in enumerate(candidates) if position != index]
        for other in others:
            value = cosine_similarity(probe, other.vector_full)
            best_other = max(best_other, value)
        if genuine_similarity > best_other:
            rank1_hits += 1

        # Amostra de impostores para as curvas de FAR/FRR.
        sample = rng.sample(others, min(max_impostors, len(others)))
        for other in sample:
            impostor.append(cosine_similarity(probe, other.vector_full))

    rank1_accuracy = rank1_hits / len(candidates) if candidates else 0.0
    return genuine, impostor, rank1_accuracy


def far(impostor: list[float], threshold: float) -> float:
    """Taxa de falsa aceitacao: impostores com similaridade acima do limiar."""

    if not impostor:
        return 0.0
    return sum(1 for value in impostor if value >= threshold) / len(impostor)


def frr(genuine: list[float], threshold: float) -> float:
    """Taxa de falsa rejeicao: genuinos com similaridade abaixo do limiar."""

    if not genuine:
        return 0.0
    return sum(1 for value in genuine if value < threshold) / len(genuine)


def equal_error_rate(
    genuine: list[float],
    impostor: list[float],
) -> tuple[float, float]:
    """Varre limiares e devolve ``(EER, threshold no EER)``.

    O EER e o ponto em que a falsa aceitacao e a falsa rejeicao se igualam —
    uma medida resumida da qualidade do separador, independente de threshold.
    """

    best_threshold = 0.0
    best_gap = 2.0
    best_eer = 1.0
    steps = 400
    for step in range(steps + 1):
        threshold = -1.0 + 2.0 * step / steps
        current_far = far(impostor, threshold)
        current_frr = frr(genuine, threshold)
        gap = abs(current_far - current_frr)
        if gap < best_gap:
            best_gap = gap
            best_eer = (current_far + current_frr) / 2.0
            best_threshold = threshold
    return best_eer, best_threshold


def percentile(values: list[float], fraction: float) -> float:
    """Valor no percentil pedido (``fraction`` entre 0 e 1)."""

    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


def precision_recall(
    genuine: list[float],
    impostor: list[float],
    threshold: float,
) -> tuple[float, float]:
    """Precisao e revocacao no threshold dado."""

    true_positive = sum(1 for value in genuine if value >= threshold)
    false_negative = sum(1 for value in genuine if value < threshold)
    false_positive = sum(1 for value in impostor if value >= threshold)
    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0.0
    )
    return precision, recall


def evaluate_scenario(
    scenario: str,
    probe_variant: str,
    candidates: list[Candidate],
    rng: random.Random,
    max_impostors: int,
) -> ScenarioReport:
    """Calcula todas as metricas de um cenario."""

    genuine, impostor, rank1 = collect_pairs(
        candidates, probe_variant, rng, max_impostors
    )
    # Cenario degenerado: o probe e o proprio embedding completo, entao o par
    # genuino e trivialmente perfeito. Marcado para o leitor nao se enganar.
    degenerate = probe_variant == "full"

    eer, eer_threshold = equal_error_rate(genuine, impostor)
    # Threshold sugerido: logo acima de quase todo impostor, para manter a
    # falsa aceitacao baixa.
    suggested = round(min(0.95, percentile(impostor, 0.99) + 0.02), 4)
    precision, recall = precision_recall(genuine, impostor, suggested)

    return ScenarioReport(
        scenario=scenario,
        probe_variant=probe_variant,
        degenerate=degenerate,
        rank1_accuracy=round(rank1, 4),
        genuine_count=len(genuine),
        impostor_count=len(impostor),
        genuine_mean=round(sum(genuine) / len(genuine), 4) if genuine else 0.0,
        impostor_mean=round(sum(impostor) / len(impostor), 4) if impostor else 0.0,
        impostor_p99=round(percentile(impostor, 0.99), 4),
        eer=round(eer, 4),
        eer_threshold=round(eer_threshold, 4),
        suggested_threshold=suggested,
        far_at_suggested=round(far(impostor, suggested), 4),
        frr_at_suggested=round(frr(genuine, suggested), 4),
        precision_at_suggested=round(precision, 4),
        recall_at_suggested=round(recall, 4),
    )


def build_report(candidates: list[Candidate], max_impostors: int) -> dict[str, Any]:
    """Monta o relatorio completo, com um cenario por caminho do pipeline."""

    rng = random.Random(SEED)
    scenarios = [
        evaluate_scenario(scenario, variant, candidates, rng, max_impostors)
        for scenario, variant in SCENARIOS.items()
    ]
    return {
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "gallery_size": len(candidates),
        "max_impostors_per_identity": max_impostors,
        "seed": SEED,
        "note": (
            "Avaliacao sintetica a partir do manifesto. Limite superior "
            "otimista; para metricas reais use um conjunto de probes."
        ),
        "scenarios": [scenario.__dict__ for scenario in scenarios],
    }


def print_summary(report: dict[str, Any]) -> None:
    """Imprime um resumo legivel do relatorio no terminal."""

    print(f"galeria: {report['gallery_size']} procurados")
    print(f"modelo: {report['embedding_model']}")
    print()
    for scenario in report["scenarios"]:
        print(f"[{scenario['scenario']}] probe={scenario['probe_variant']}")
        if scenario["degenerate"]:
            print(
                "  aviso: probe igual a galeria, genuino trivial. Use probes reais."
            )
        print(
            f"  rank-1={scenario['rank1_accuracy'] * 100:.1f}%  "
            f"EER={scenario['eer'] * 100:.1f}% @ thr={scenario['eer_threshold']}"
        )
        print(
            f"  genuino_med={scenario['genuine_mean']}  "
            f"impostor_med={scenario['impostor_mean']}  "
            f"impostor_p99={scenario['impostor_p99']}"
        )
        print(
            f"  threshold sugerido={scenario['suggested_threshold']}  "
            f"FAR={scenario['far_at_suggested'] * 100:.2f}%  "
            f"FRR={scenario['frr_at_suggested'] * 100:.2f}%"
        )
        print()
