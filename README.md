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

## Observações de privacidade

Vetor facial é dado biométrico derivado de pessoa natural. Mesmo partindo de
página pública, mantenha o bucket privado, restrinja chaves de acesso, evite
logs com dados pessoais além do necessário e documente a finalidade acadêmica
do tratamento.
