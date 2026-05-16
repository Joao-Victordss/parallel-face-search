# Sync MJ Procurados

Sincroniza diariamente a lista pública de procurados do Ministério da Justiça
e Segurança Pública, gera vetor facial a partir da imagem pública e grava os
dados em Cloudflare R2.

O dataset gravado contém apenas:

- nome;
- estado de origem;
- data de inclusão na listagem;
- data de atualização da página pública;
- link da página pública;
- vetor facial gerado a partir da imagem.

As imagens não são gravadas no R2. Elas são baixadas apenas durante a execução para gerar o vetor.

## Estrutura no R2

Por padrão, os objetos ficam no prefixo `mj-procurados/`:

```text
mj-procurados/manifest.json
mj-procurados/records/<id>.json
```

O `manifest.json` contém todos os registros e seus vetores. Os arquivos em
`records/` existem para consulta individual.

## Configuração

Crie um bucket no Cloudflare R2 e gere uma chave S3 com permissão de
leitura/escrita apenas nesse bucket.

No repositório GitHub, configure estes secrets:

```text
R2_ACCOUNT_ID
R2_ACCESS_KEY_ID
R2_SECRET_ACCESS_KEY
R2_BUCKET
```

O workflow em `.github/workflows/sync-mj-procurados.yml` roda uma vez por dia
às `06:00 UTC` e também pode ser executado manualmente em `workflow_dispatch`.

## Rodando localmente

Instale as dependências:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

No Windows, prefira deixar o projeto em um caminho sem acentos e sem caracteres
especiais, por exemplo `C:\dev\parallel-face-search`. O `dlib`, usado pelo
`face_recognition`, pode falhar ao abrir os modelos `.dat` quando o caminho tem
caracteres não ASCII.

Para validar o ambiente:

```bash
python scripts/check_environment.py
```

Teste sem enviar para o R2:

```bash
python scripts/sync_mj_procurados.py --no-upload --limit 3
```

Com variáveis R2 configuradas, rode:

```bash
python scripts/sync_mj_procurados.py
```

Sem `--no-upload`, o script exige `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`,
`R2_SECRET_ACCESS_KEY` e `R2_BUCKET`. Isso evita que o GitHub Actions termine
como sucesso gravando apenas em arquivo local quando o R2 estiver mal
configurado.

## Comparação Pela Webcam

O script `scripts/webcam_face_search.py` abre a webcam, gera o vetor facial do
rosto detectado no vídeo e compara contra os vetores salvos no `manifest.json`.

Para rodar carregando o manifesto direto do R2:

```bash
set -a
source .env
set +a
python scripts/webcam_face_search.py --mode parallel --workers 4
```

Para rodar com um `manifest.json` local:

```bash
python scripts/webcam_face_search.py \
  --manifest out/mj-procurados/manifest.json \
  --mode parallel \
  --workers 4
```

Modos disponíveis:

- `sequential`: compara o rosto da webcam contra a base em um único processo.
- `parallel`: divide a base entre múltiplos workers/processos.
- `benchmark`: executa as duas estratégias e mostra tempo médio e speedup.

Exemplo para análise de desempenho:

```bash
python scripts/webcam_face_search.py --mode benchmark --workers 4 --repeat 100
```

O `--repeat` repete a mesma comparação para aumentar a carga computacional e
deixar a diferença entre sequencial e paralelo mais visível no relatório.
Durante a execução, pressione `q` para sair.

O percentual exibido na tela é um score derivado da distância entre vetores, não
uma probabilidade estatística calibrada. Use como métrica de similaridade.

Se a câmera não abrir, liste os índices disponíveis:

```bash
python scripts/webcam_face_search.py --list-cameras
```

Depois rode informando o índice encontrado:

```bash
python scripts/webcam_face_search.py --mode benchmark --camera 1
```

No Linux, também confira se existe algum dispositivo `/dev/video*`:

```bash
ls -l /dev/video*
```

Se estiver usando WSL e não aparecer nenhum `/dev/video*`, a webcam não está
exposta para o Linux. Nesse caso, rode o projeto em Linux nativo ou no Python do
Windows, ou conecte a câmera ao WSL antes de executar o script.

## Observações de privacidade

Vetor facial é dado biométrico derivado de pessoa natural. Mesmo partindo de
página pública, mantenha o bucket privado, restrinja chaves de acesso, evite
logs com dados pessoais além do necessário e documente a finalidade acadêmica
do tratamento.
