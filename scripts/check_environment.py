from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path


def check_import(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        print(f"ok: {module_name}")
        return True
    except Exception as exc:
        print(f"erro: {module_name}: {exc}")
        return False


def has_non_ascii_path(path: Path) -> bool:
    try:
        str(path).encode("ascii")
        return False
    except UnicodeEncodeError:
        return True


def main() -> int:
    print(f"python: {sys.version.split()[0]}")
    print(f"sistema: {platform.platform()}")
    print(f"cwd: {Path.cwd()}")

    if platform.system() == "Windows" and has_non_ascii_path(Path.cwd()):
        print(
            "aviso: o caminho do projeto tem caracteres nao ASCII. "
            "O dlib/face_recognition pode falhar ao abrir os modelos .dat no Windows. "
            "Prefira mover o projeto para algo como C:\\dev\\parallel-face-search."
        )

    modules = [
        "cv2",
        "numpy",
        "psutil",
        "boto3",
        "face_recognition_models",
        "face_recognition",
    ]
    results = [check_import(module) for module in modules]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
