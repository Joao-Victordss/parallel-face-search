# Windows Setup

Use um caminho simples, sem acentos, para evitar falhas do `dlib` ao abrir os
modelos do `face_recognition`.

## 1. Clonar em `C:\dev`

No PowerShell:

```powershell
cd C:\
mkdir dev
cd C:\dev
git clone git@github.com:Joao-Victordss/parallel-face-search.git
cd parallel-face-search
```

## 2. Criar ambiente virtual

Use Python 3.10 ou 3.11.

```powershell
py -3.10 -m venv .venv-win
.\.venv-win\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Se `py -3.10` não existir, tente:

```powershell
py -m venv .venv-win
```

## 3. Configurar R2

Crie um arquivo `.env` na raiz do projeto:

```text
R2_ACCOUNT_ID=seu-account-id
R2_ACCESS_KEY_ID=sua-access-key
R2_SECRET_ACCESS_KEY=sua-secret-key
R2_BUCKET=mj-procurados
R2_PREFIX=mj-procurados
```

Carregue o `.env` no PowerShell:

```powershell
Get-Content .env | ForEach-Object {
  if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
    [Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), 'Process')
  }
}
```

## 4. Validar ambiente

```powershell
python scripts\check_environment.py
python scripts\webcam_face_search.py --list-cameras
```

O `check_environment.py` deve imprimir `ok` para todos os módulos.

## 5. Rodar comparação

```powershell
python scripts\webcam_face_search.py --mode benchmark --workers 4 --repeat 100
```

Se houver mais de uma câmera:

```powershell
python scripts\webcam_face_search.py --mode benchmark --workers 4 --repeat 100 --camera 1
```

## Observação

O percentual exibido é um score derivado da distância entre vetores, não uma
probabilidade estatística calibrada. Para análise, use principalmente distância,
tempo médio, speedup e uso de CPU.
