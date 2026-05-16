"""Comando ``face-search-check``: valida o ambiente de execucao.

Confere se as dependencias estao instaladas, se o pacote ``face_search`` esta
importavel e se o provider CUDA do ONNX Runtime esta disponivel. Util como
primeiro passo apos a instalacao.
"""

from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path


def check_import(module_name: str) -> bool:
    """Tenta importar um modulo e imprime o resultado."""

    try:
        importlib.import_module(module_name)
        print(f"ok: {module_name}")
        return True
    except Exception as exc:
        print(f"erro: {module_name}: {exc}")
        return False


def check_onnx_providers() -> None:
    """Lista os providers do ONNX Runtime e avisa se a GPU nao estiver disponivel."""

    try:
        import onnxruntime
    except Exception:
        return

    providers = onnxruntime.get_available_providers()
    print(f"onnxruntime providers: {', '.join(providers)}")
    if "CUDAExecutionProvider" not in providers:
        print(
            "aviso: CUDAExecutionProvider indisponivel. A webcam roda em CPU, "
            "mais lenta. Para GPU, instale o requirements-gpu.txt e confira o "
            "driver CUDA. O sync funciona normalmente em CPU."
        )


def main() -> int:
    """Roda todas as verificacoes. Devolve 0 se tudo passou, 1 caso contrario."""

    print(f"python: {sys.version.split()[0]}")
    print(f"sistema: {platform.platform()}")
    print(f"cwd: {Path.cwd()}")

    modules = [
        "cv2",
        "numpy",
        "boto3",
        "onnxruntime",
        "insightface",
        "face_search",
    ]
    results = [check_import(module) for module in modules]
    check_onnx_providers()
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
