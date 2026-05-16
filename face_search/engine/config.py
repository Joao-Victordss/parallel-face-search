"""Configuracao do motor facial.

O ``EngineConfig`` reune todos os parametros que controlam a deteccao e a
codificacao de rostos. O mesmo objeto e usado pelo pipeline de sincronizacao
(galeria) e pelo pipeline da webcam, garantindo que a galeria e os rostos
capturados sejam processados exatamente com os mesmos parametros.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class EngineConfig:
    """Parametros do motor de deteccao e codificacao facial.

    O objeto e imutavel (``frozen=True``) de proposito: depois de criado, a
    configuracao nao muda durante a execucao, o que evita inconsistencias
    entre a galeria e o probe.
    """

    # Provider de execucao do ONNX Runtime. "cuda" usa a GPU NVIDIA;
    # "cpu" usa apenas o processador (mais lento, suficiente para o sync).
    onnx_provider: str = "cuda"

    # Pacote de modelos do InsightFace. "buffalo_l" traz o detector SCRFD e
    # o codificador ArcFace de 512 dimensoes.
    insightface_pack: str = "buffalo_l"

    # Lado, em pixels, da imagem quadrada usada pelo detector SCRFD.
    det_size: int = 640

    # Menor lado aceitavel da caixa do rosto, em pixels. Rostos menores que
    # isso sao considerados pequenos demais para gerar um embedding confiavel.
    min_face: int = 40

    # Limiar de foco (variancia do Laplaciano). Esta deixado baixo de
    # proposito: o gate de qualidade so deve barrar imagem realmente quebrada.
    # Uma foto oficial mediana ainda gera embedding util, e frames borrados ja
    # sao ponderados de forma continua pelo acumulo de evidencia.
    min_focus: float = 12.0

    # Faixa de brilho medio aceitavel (0 a 255). Fora dela, o recorte esta
    # escuro ou estourado demais.
    min_brightness: float = 25.0
    max_brightness: float = 245.0

    @classmethod
    def from_env(cls) -> "EngineConfig":
        """Cria a configuracao a partir de variaveis de ambiente.

        Usado pelo pipeline de sincronizacao, que roda no GitHub Actions e
        recebe os parametros pelo ambiente do workflow.
        """

        return cls(
            onnx_provider=os.getenv("ONNX_PROVIDER", "cuda").strip().lower(),
            insightface_pack=os.getenv("INSIGHTFACE_PACK", "buffalo_l").strip(),
            det_size=int(os.getenv("DET_SIZE", "640")),
        )

    def providers(self) -> list[str]:
        """Lista de providers do ONNX Runtime, na ordem de preferencia.

        Com CUDA, o provider de GPU vem primeiro e o de CPU fica como reserva.
        """

        if self.onnx_provider == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def ctx_id(self) -> int:
        """Identificador do dispositivo para o InsightFace.

        ``0`` seleciona a primeira GPU; ``-1`` forca a execucao na CPU.
        """

        return 0 if self.onnx_provider == "cuda" else -1
