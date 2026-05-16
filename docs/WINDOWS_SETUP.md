# Guia de instalação no Windows

Passo a passo para rodar o projeto no Windows, incluindo a webcam com GPU
NVIDIA.

## 1. Clonar o repositório

No PowerShell:

```powershell
cd C:\
mkdir dev
cd C:\dev
git clone git@github.com:Joao-Victordss/parallel-face-search.git
cd parallel-face-search
```

## 2. Criar o ambiente virtual

Use Python 3.10 ou 3.11.

```powershell
py -3.10 -m venv .venv-win
.\.venv-win\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
```

Se `py -3.10` não existir, tente `py -m venv .venv-win`.

## 3. Instalar o pacote `face_search`

O projeto é um pacote Python instalável. A instalação em modo editável
(`-e`) deixa o código editável e cria os comandos `face-search-*`:

```powershell
python -m pip install -e .
```

> Se a janela da webcam não abrir mais tarde (erro de `cv2.imshow`), é porque
> o `insightface` puxou o `opencv-python-headless`, que não tem interface
> gráfica. Corrija mantendo apenas o pacote com GUI:
>
> ```powershell
> python -m pip uninstall -y opencv-python opencv-python-headless
> python -m pip install --no-cache-dir opencv-python==4.10.0.84
> ```

## 4. Dependências de GPU

A webcam roda melhor com GPU NVIDIA. Para isso, instale também o
`requirements-gpu.txt`. Antes, confira se o driver NVIDIA está instalado
rodando `nvidia-smi` no PowerShell.

```powershell
python -m pip install -r requirements-gpu.txt
```

O `onnxruntime-gpu` precisa de uma versão compatível de CUDA e cuDNN
instalada na máquina. Sem GPU, o sistema ainda funciona em CPU, apenas mais
lento; nesse caso pule este passo e rode a webcam com `--onnx-provider cpu`.

## 5. Configurar o R2

Crie um arquivo `.env` na raiz do projeto, a partir do `.env.example`:

```text
R2_ACCOUNT_ID=seu-account-id
R2_ACCESS_KEY_ID=sua-access-key
R2_SECRET_ACCESS_KEY=sua-secret-key
R2_BUCKET=mj-procurados
R2_PREFIX=mj-procurados
ONNX_PROVIDER=cuda
```

Carregue o `.env` no PowerShell:

```powershell
Get-Content .env | ForEach-Object {
  if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
    [Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), 'Process')
  }
}
```

## 6. Validar o ambiente

```powershell
face-search-check
face-search-webcam --list-cameras
```

O `face-search-check` deve imprimir `ok` para todos os módulos. Ele também
lista os providers do ONNX Runtime. Se `CUDAExecutionProvider` não aparecer,
a webcam roda em CPU.

## 7. Rodar a comparação

```powershell
face-search-webcam --mode benchmark --workers 4 --repeat 100
```

Se houver mais de uma câmera, informe o índice:

```powershell
face-search-webcam --mode benchmark --workers 4 --camera 1
```

## Observação

A confiança exibida na tela é o resultado do acúmulo de evidência ao longo
dos frames, não uma probabilidade estatística calibrada. Para análise de
desempenho do paralelismo, use o tempo médio de cada modo e o ganho de
velocidade mostrados no resumo final.
