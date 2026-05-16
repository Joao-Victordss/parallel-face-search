# Roadmap de Acurácia e Reconhecimento com Máscara

> Status: implementado. O sistema descrito neste roadmap foi construído. A
> documentação operacional vigente é o `README.md`. Em relação a este plano
> inicial, a versão implementada distingue **três** tipos de máscara em vez de
> dois (nenhuma, parcial e ocular) e guarda **três** variantes de embedding
> por procurado (`full`, `upper`, `periocular`) no campo `embeddings` do
> manifesto. O classificador de tipo de máscara começou heurístico, com
> modelo treinado previsto como evolução. O detalhamento da implementação
> está no arquivo de plano em `~/.claude/plans/`.

Plano estruturado para que o projeto detecte rostos de pessoas procuradas
pela justiça **com alta acurácia, inclusive quando a pessoa usa máscara**.

## Contexto e limite honesto

A galeria de procurados do MJ são fotos **sem máscara**. A webcam pode
capturar um rosto **com máscara**. Comparar um rosto mascarado contra uma
galeria sem máscara é um problema reconhecidamente difícil — estudos do NIST
(FRVT) mostram queda de acurácia mesmo em modelos estado-da-arte.

Consequências assumidas neste roadmap:

- O resultado é sempre um **candidato para verificação humana**, nunca uma
  confirmação automática de identidade.
- O caminho com máscara terá teto de acurácia menor que o sem máscara; cada
  caminho é calibrado e avaliado separadamente.

## Pré-requisito — troca do motor de reconhecimento

O motor atual (`face_recognition`/dlib) **não funciona com máscara**: o
embedding 128d depende da metade inferior do rosto (nariz, boca, queixo),
exatamente o que a máscara cobre. Não é ajuste de parâmetro — o modelo muda.

Novo motor: **InsightFace** sobre GPU CUDA.

| Componente | Antes (dlib) | Depois (InsightFace) |
|------------|--------------|----------------------|
| Detecção | `face_locations` hog/cnn | SCRFD (detecta rostos mascarados) |
| Reconhecimento | encoding 128d | ArcFace, embedding 512d (`buffalo_l`) |
| Execução | CPU | `onnxruntime-gpu` (CUDA) |
| Métrica | distância euclidiana | similaridade de cosseno |

**Impacto da migração** (tarefa transversal, fazer primeiro):

- Embedding muda de 128d → 512d. `load_candidates` valida `len == 128` hoje.
- Formato do `manifest.json` muda (ver Frente 3). `FACE_VECTOR_MODEL` muda.
- O `sync` inteiro precisa ser re-executado para regerar a galeria.
- `requirements.txt`: adicionar `insightface` e `onnxruntime-gpu`; remover
  `face-recognition` / `face-recognition-models` / `dlib`.
- O paralelismo (`sequential`/`parallel`/`benchmark`) é **mantido** — a
  camada de comparação continua, agora sobre vetores 512d com cosseno.

---

## Pipeline alvo

```
scrape MJ ─► detecta rosto (SCRFD) ─► detecta máscara? ─┬─ não ─► embedding rosto completo (ArcFace)
                                                        └─ sim ─► embedding caminho-máscara
                                                      ─► comparação paralela (cosseno) ─► candidato
```

A galeria guarda **dois embeddings por procurado**; a webcam escolhe contra
qual comparar conforme a detecção de máscara.

---

## Frente 1 — Detecção de rosto

**Subagente:** `face-detection`

- Substituir `face_locations` (hog/cnn) por **SCRFD** do InsightFace, que
  detecta rostos com e sem máscara.
- Usar o provider CUDA do `onnxruntime`.
- Aproveitar os 5 pontos do SCRFD para alinhar a face antes de codificar.
- Manter gate de qualidade (variância do Laplaciano p/ foco, brilho médio) e
  tamanho mínimo de rosto.

---

## Frente 2 — Detecção de máscara (NOVA)

**Subagente:** `mask-detection`

- Classificador binário **com / sem máscara** aplicado a cada rosto detectado.
- É o roteador do pipeline adaptativo: define qual embedding gerar e contra
  qual embedding da galeria comparar.
- Modelo leve em ONNX (classificador MobileNet de máscara) rodando na GPU.
- A janela da webcam passa a exibir o estado da máscara por rosto.
- Erro deste classificador propaga para o resto — calibrar o limiar dele e
  medir no harness (cenário dedicado).

---

## Frente 3 — Geração de embedding

**Subagente:** `face-embedding`

- **ArcFace 512d** com alinhamento pelos 5 pontos do SCRFD.
- **Galeria (sync) — dois embeddings por procurado:**
  - `face_vector_full`: rosto completo da foto oficial.
  - `face_vector_masked`: aplica **máscara sintética** sobre a foto oficial
    (renderização por landmarks) e re-codifica com o mesmo ArcFace. Assim o
    probe mascarado e a galeria ficam no mesmo domínio — é o que sustenta a
    acurácia do caminho com máscara.
- **Probe (webcam):** gera um embedding; o caminho (full/masked) vem da
  Frente 2.
- TTA leve (flip horizontal) para estabilizar o vetor.
- Manifesto versiona modelo, dimensão e parâmetros (ver formato abaixo).

Formato novo de registro:

```json
{
  "id": "...",
  "name": "...",
  "state": "...",
  "face_vector_full":   [512 floats],
  "face_vector_masked": [512 floats],
  "embedding_model": "insightface/arcface-buffalo_l-512d",
  "synthetic_mask_model": "<ferramenta>@<versão>"
}
```

`can_reuse_embedding` deve comparar também o modelo/params — embeddings de
versões diferentes não podem coexistir no manifesto.

---

## Frente 4 — Matching adaptativo e threshold

**Subagente:** `face-matching`

- Probe **sem máscara** → compara contra `face_vector_full`.
- Probe **com máscara** → compara contra `face_vector_masked`.
- Métrica passa a **similaridade de cosseno** (embeddings ArcFace
  normalizados → cosseno = produto interno).
- **Dois thresholds calibrados separadamente** pela Frente 5; o do caminho
  com máscara é mais conservador.
- Teste de razão (gap entre top-1 e top-2) e voto temporal entre frames
  para reduzir falsos positivos.
- Toda exibição deixa claro que é candidato para verificação humana.
- **Paralelismo mantido:** os modos `sequential`/`parallel`/`benchmark`
  continuam; a função de comparação didática passa a calcular cosseno.

---

## Frente 5 — Harness de avaliação

**Subagente:** `accuracy-eval`

- Harness em `face_search/evaluation.py`, exposto pelo comando
  `face-search-evaluate`.
- Avaliar em **três cenários separados**: sem máscara, com máscara sintética,
  com máscara real (quando houver amostras).
- Métricas por cenário: FAR, FRR, EER, precisão/revocação, acurácia rank-1.
- Avaliar também o classificador de máscara da Frente 2 isoladamente.
- Varredura que produz os **dois thresholds** da Frente 4.
- Relatório JSON comparável entre execuções, com os parâmetros do modelo.

---

## Ordem de execução

1. **Migração de motor** — InsightFace + GPU, atualizar `requirements.txt` e
   `load_candidates` (512d).
2. **Frente 1 + Frente 3 (básico)** — regerar a galeria com `face_vector_full`.
3. **Frente 5** — harness; sem medição não há "alta acurácia" comprovável.
4. **Frente 2** — detecção de máscara.
5. **Frente 3 (máscara)** — máscara sintética + `face_vector_masked`.
6. **Frente 4** — matching adaptativo e calibração dos dois thresholds.
7. Iterar 1-4 medindo cada mudança pela Frente 5.

## Subagentes por frente

| Frente | Subagente | Arquivo |
|--------|-----------|---------|
| 1 | `face-detection` | `.claude/agents/face-detection.md` |
| 2 | `mask-detection` | `.claude/agents/mask-detection.md` |
| 3 | `face-embedding` | `.claude/agents/face-embedding.md` |
| 4 | `face-matching` | `.claude/agents/face-matching.md` |
| 5 | `accuracy-eval` | `.claude/agents/accuracy-eval.md` |

## Skills alinhadas

- **`security-review`** — vetor facial é dado biométrico de pessoa natural
  em contexto de identificação criminal (LGPD). Rodar a cada mudança em
  coleta, armazenamento ou logs.
- **`simplify`** — revisar o código alterado nas Frentes 1-4 sem regredir a
  versão didática do paralelismo.
- **`review`** — revisão de PR ao consolidar cada frente.
