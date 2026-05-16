"""Parallel Face Search — reconhecimento facial de procurados pela webcam.

Este pacote identifica pessoas da lista publica de procurados do Ministerio
da Justica comparando rostos capturados pela webcam contra uma galeria de
vetores faciais. O reconhecimento funciona mesmo quando a pessoa usa mascara,
porque cada rosto e comparado por tres caminhos independentes (rosto exposto,
metade superior e faixa dos olhos).

O codigo esta dividido em subpacotes, cada um com uma responsabilidade clara:

- ``engine``   — deteccao, alinhamento, codificacao (embedding) e qualidade.
- ``gallery``  — montagem da galeria: scraping do MJ, manifesto e Cloudflare R2.
- ``matching`` — comparacao por cosseno e busca paralela contra a galeria.
- ``tracking`` — rastreamento de rostos e acumulo de evidencia entre frames.
- ``webcam``   — pipeline da webcam: camera, HUD e laco de reconhecimento.
- ``cli``      — pontos de entrada de linha de comando.

O resultado deste sistema e sempre um candidato para verificacao humana,
nunca uma confirmacao automatica de identidade.
"""

from __future__ import annotations

__version__ = "1.0.0"
