# Parallel Face Search

Sistema que identifica pessoas da lista pública de procurados do Ministério
da Justiça e Segurança Pública comparando rostos capturados pela webcam
contra uma galeria de vetores faciais. O sistema reconhece rostos mesmo
quando a pessoa usa máscara, comparando cada rosto de três formas separadas
e somando a evidência das três numa única confiança.

> ⚠️ **Aviso importante** — O resultado deste sistema é sempre um *candidato
> para verificação humana*, nunca uma confirmação automática de identidade.
> Reconhecimento facial de rostos parcialmente cobertos é um problema difícil
> e sujeito a erro. Use a saída como apoio à investigação, não como prova.

## Índice

- [Objetivo](#objetivo)
- [Como funciona](#como-funciona)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Instalação](#instalação)
- [Configuração do Cloudflare R2](#configuração-do-cloudflare-r2)
- [Como usar](#como-usar)
- [Modos de comparação e paralelismo](#modos-de-comparação-e-paralelismo)
- [Privacidade](#privacidade)

## Objetivo

A lista de procurados é pública, mas só traz a foto oficial de cada pessoa.
Em situações reais a pessoa pode estar usando máscara cirúrgica, balaclava ou
touca. O sistema resolve isso de duas formas:

1. Para cada procurado, gera três versões do vetor facial: rosto completo,
   rosto com a metade inferior coberta e rosto com tudo coberto exceto os
   olhos.
2. Na webcam, compara cada rosto contra as três versões da galeria, sem
   tentar adivinhar o tipo de máscara.

## Como funciona

### Os três caminhos de comparação

Em vez de detectar o tipo de máscara e escolher um caminho, o sistema compara
cada rosto pelos três caminhos, sempre, de forma separada. Assim não depende
de acertar uma classificação de máscara.

| Caminho       | Região do rosto usada      | Versão da galeria          | Peso |
|---------------|----------------------------|----------------------------|------|
| Face exposta  | rosto inteiro              | `full` (rosto completo)    | 1.00 |
| Face coberta  | metade superior do rosto   | `upper` (metade superior)  | 0.60 |
| Olhos         | faixa dos olhos            | `periocular` (faixa olhos) | 0.35 |

A face exposta vale mais, porque uma região maior é mais confiável. Se a
pessoa está sem máscara, os três caminhos casam e somam evidência rápido. Se
está com máscara, o caminho da face exposta não casa, e só os caminhos
compatíveis somam, mais devagar.

### Acúmulo de evidência

A análise de um único frame de vídeo é ruidosa, ainda mais com máscara. Por
isso o sistema não decide a identidade em um frame só. Ele acompanha cada
rosto ao longo dos frames e soma a evidência de cada comparação:

1. Um rastreador liga o rosto a um identificador estável entre frames.
2. A cada frame, os três caminhos de comparação rodam. Cada caminho que
   encontra um candidato soma evidência ao acumulador daquele rosto.
3. Frames e caminhos ruidosos (foco ruim, máscara, detecção fraca)
   contribuem com peso menor.
4. A confiança exibida cresce conforme frames consistentes apontam para o
   mesmo candidato. Um rosto mascarado exige naturalmente mais frames.
5. Se um rosto sai da cena e volta, o sistema o re-identifica pelo próprio
   embedding e a confiança continua de onde parou, em vez de zerar.

### Os dois pipelines

O projeto tem dois fluxos: um monta a galeria, o outro consulta.

**Pipeline de sincronização (galeria)** — roda no GitHub Actions uma vez por
dia e também pode rodar localmente. Faz o scraping da lista pública, baixa a
foto oficial de cada procurado (apenas em memória, nunca gravada), detecta e
alinha o rosto com o SCRFD, gera três vetores de 512 dimensões com o ArcFace
e grava os vetores e metadados no Cloudflare R2.

**Pipeline da webcam (consulta)** — roda na máquina local, de preferência com
GPU NVIDIA. Captura um frame, detecta e alinha cada rosto, rastreia, recorta
as três regiões, gera os três vetores numa única inferência, compara contra a
galeria por similaridade de cosseno, acumula a evidência e desenha o
resultado na tela.

## Estrutura do projeto

O código é um pacote Python instalável, `face_search`, dividido em
subpacotes com responsabilidades claras:

```text
face_search/
├── engine/        Detecção, alinhamento, codificação (embedding) e qualidade
│   ├── config.py      EngineConfig — parâmetros do motor
│   ├── detection.py   Detecção e alinhamento de rostos (SCRFD)
│   ├── embedding.py   Codificação em vetores 512d (ArcFace)
│   ├── quality.py     Gate de qualidade do recorte facial
│   └── regions.py     Extração das 3 regiões (full/upper/periocular)
├── gallery/       Montagem da galeria
│   ├── scraper.py     Scraping da lista pública do MJ
│   ├── builder.py     Geração dos embeddings e orquestração do sync
│   ├── manifest.py    Formato, validação e cache do manifest.json
│   └── r2.py          Configuração e acesso ao Cloudflare R2
├── matching/      Comparação contra a galeria
│   ├── similarity.py  Similaridade de cosseno (Python puro, didático)
│   ├── candidate.py   Candidate/Match e carregamento do manifesto
│   └── search.py      Busca dos melhores candidatos, sequencial e paralela
├── tracking/      Continuidade temporal
│   ├── tracker.py     FaceTracker — liga um rosto a um track_id estável
│   └── evidence.py    EvidenceAccumulator — acúmulo de evidência por frame
├── webcam/        Pipeline da webcam
│   ├── camera.py      Descoberta e abertura da webcam
│   ├── hud.py         Interface sobreposta ao vídeo
│   └── pipeline.py    Laço principal de reconhecimento
├── evaluation.py  Harness de avaliação de acurácia
└── cli/           Pontos de entrada de linha de comando
```

Cada comando `face-search-*` corresponde a um módulo em `cli/`:

| Comando                     | O que faz                                          |
|-----------------------------|----------------------------------------------------|
| `face-search-check`         | Valida o ambiente e as dependências.               |
| `face-search-sync`          | Sincroniza a galeria de procurados para o R2.      |
| `face-search-webcam`        | Busca facial pela webcam.                          |
| `face-search-build-gallery` | Monta uma galeria a partir de imagens locais.      |
| `face-search-match`         | Testa o reconhecimento com fotos em disco.         |
| `face-search-evaluate`      | Avalia a acurácia e sugere thresholds.             |

Tecnologias: detector **SCRFD** e codificador **ArcFace** (512d) via
**InsightFace**; comparação por **similaridade de cosseno**; armazenamento da
galeria no **Cloudflare R2**.

## Instalação

Use Python **3.10 ou 3.11** (as dependências de visão computacional ainda não
suportam versões mais novas).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

O comando `pip install -e .` instala o pacote `face_search` em modo editável
e cria os comandos `face-search-*`. Ele instala o `onnxruntime` de CPU, que é
suficiente para a sincronização da galeria.

Na máquina que roda a webcam com **GPU NVIDIA**, instale também as
dependências de GPU:

```bash
pip install -r requirements-gpu.txt
```

Valide o ambiente:

```bash
face-search-check
```

No Windows, siga o guia em [`docs/WINDOWS_SETUP.md`](docs/WINDOWS_SETUP.md).

### A janela da webcam não abre

Se a webcam falhar com `cv2.imshow ... function is not implemented`, o `cv2`
ativo é a versão *headless* (sem janela gráfica), que o `insightface` puxa
como dependência. Mantenha apenas o `opencv-python` com interface:

```bash
pip uninstall -y opencv-python opencv-python-headless
pip install --no-cache-dir opencv-python==4.10.0.84
```

## Configuração do Cloudflare R2

A galeria de vetores faciais é gravada no Cloudflare R2. Crie um bucket e gere
uma chave S3 com permissão de leitura e escrita apenas nesse bucket. Configure
os secrets no repositório GitHub:

```text
R2_ACCOUNT_ID
R2_ACCESS_KEY_ID
R2_SECRET_ACCESS_KEY
R2_BUCKET
```

Para rodar localmente, crie um arquivo `.env` na raiz a partir do
[`.env.example`](.env.example).

## Como usar

### Sincronizar a galeria

Teste local sem enviar ao R2 (grava em `out/`):

```bash
face-search-sync --no-upload --limit 3
```

Com as variáveis R2 configuradas, sincroniza e envia ao R2:

```bash
face-search-sync
```

Sem `--no-upload`, o comando exige as quatro variáveis R2. O workflow em
`.github/workflows/sync-mj-procurados.yml` roda uma vez por dia às 06:00 UTC
e também pode ser disparado manualmente.

### Buscar pela webcam

Carregando a galeria direto do R2:

```bash
set -a; source .env; set +a
face-search-webcam --mode parallel --workers 4
```

Com um manifesto local:

```bash
face-search-webcam \
  --manifest out/mj-procurados/manifest.json \
  --mode parallel --workers 4
```

Pressione `q` para sair. Se a câmera não abrir, liste os índices e informe o
encontrado com `--camera`:

```bash
face-search-webcam --list-cameras
```

### Testar sem webcam

Para experimentar o reconhecimento sem câmera nem scraping, use fotos em
disco. Coloque imagens na pasta `amostras/` (ignorada pelo Git) e rode:

```bash
face-search-match
```

Ou monte uma galeria local e use-a com a webcam:

```bash
face-search-build-gallery \
  --face "Você=amostras/EU.jpeg" \
  --face "Procurado=amostras/foto.png" \
  --out out/demo/manifest.json
```

### Avaliar a acurácia

```bash
face-search-evaluate \
  --manifest out/mj-procurados/manifest.json \
  --report out/relatorio.json
```

O comando mede a qualidade da comparação para os três caminhos, calcula EER e
acurácia rank-1 e sugere os thresholds.

## Modos de comparação e paralelismo

O `face-search-webcam` tem três modos, escolhidos com `--mode`:

- `sequential` — compara o rosto contra a galeria em um único processo.
- `parallel` — divide a galeria entre vários processos.
- `benchmark` — executa os dois e mostra o tempo médio e o ganho de
  velocidade.

A função de similaridade de cosseno é escrita em **Python puro de propósito**
(ver `face_search/matching/similarity.py`). Ela é a carga de trabalho que
torna visível a diferença entre o modo sequencial e o paralelo. O parâmetro
`--repeat` repete a comparação para aumentar essa carga e tornar o ganho mais
claro no relatório:

```bash
face-search-webcam --mode benchmark --workers 4 --repeat 100
```

A confiança exibida na tela é o resultado do acúmulo de evidência ao longo
dos frames, **não** uma probabilidade estatística calibrada. Trate o valor
como uma medida de quão consistente foi a observação, e o candidato como
apoio à verificação humana.

## Privacidade

Vetor facial é dado biométrico derivado de pessoa natural. Mesmo partindo de
página pública, mantenha o bucket privado, restrinja as chaves de acesso,
evite logs com dados pessoais além do necessário e documente a finalidade do
tratamento. As imagens das fotos oficiais não são gravadas no R2: são baixadas
apenas em memória durante a sincronização para gerar os vetores. A pasta
`amostras/`, usada para testes locais, é ignorada pelo Git e nunca versionada.

### Estrutura no R2

Por padrão os objetos ficam no prefixo `mj-procurados/`:

```text
mj-procurados/manifest.json
mj-procurados/records/<id>.json
```

O `manifest.json` contém todos os registros com seus três vetores. Os
arquivos em `records/` existem para consulta individual.
