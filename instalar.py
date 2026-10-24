"""Cria um ambiente local independente e instala o Laboratório MNIST."""
import argparse
from pathlib import Path
import platform
import subprocess
import sys
import venv


BASE = Path(__file__).resolve().parent
AMBIENTE = BASE / ".venv"
PYTHON = AMBIENTE / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
VERSOES = {(3, 11), (3, 12), (3, 13)}


def executar(*argumentos):
    subprocess.run([str(argumento) for argumento in argumentos], cwd=BASE, check=True)


def verificar():
    if not PYTHON.is_file():
        raise RuntimeError("Ambiente ausente. Execute o instalador antes de iniciar o painel.")
    codigo = """
import sys
if sys.version_info[:2] not in ((3, 11), (3, 12), (3, 13)):
    raise SystemExit('Use Python 3.11, 3.12 ou 3.13; recomendamos 3.12.')
import numpy, sklearn, pandas, joblib, threadpoolctl, torch
print('Ambiente pronto: Python ' + sys.version.split()[0] + '; PyTorch ' + torch.__version__)
"""
    executar(PYTHON, "-c", codigo)
    executar(PYTHON, "-m", "pip", "check")


def instalar():
    if sys.version_info[:2] not in VERSOES:
        raise RuntimeError("Use Python 3.11, 3.12 ou 3.13; recomendamos 3.12.")
    if platform.system() not in ("Linux", "Windows", "Darwin"):
        raise RuntimeError("Este instalador atende Linux, Windows e macOS.")
    if not PYTHON.is_file():
        print("Criando ambiente .venv na pasta do projeto...", flush=True)
        venv.EnvBuilder(with_pip=True).create(AMBIENTE)
    # Confere o interpretador de um ambiente existente antes de instalar nele.
    executar(PYTHON, "-c", "import sys; "
             "assert sys.version_info[:2] in ((3,11),(3,12),(3,13)), "
             "'Recrie .venv com Python 3.11, 3.12 ou 3.13.'")
    executar(PYTHON, "-m", "pip", "install", "--upgrade", "pip")
    print("Instalando PyTorch para execução local...", flush=True)
    comando = [PYTHON, "-m", "pip", "install", "torch>=2.6,<3"]
    if platform.system() in ("Linux", "Windows"):
        comando.extend(["--index-url", "https://download.pytorch.org/whl/cpu"])
    executar(*comando)
    executar(PYTHON, "-m", "pip", "install", "-r", BASE / "requirements.txt")
    verificar()
    print("Instalação concluída. Use o inicializador do seu sistema para abrir o painel.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verificar", action="store_true",
                        help="confere o ambiente existente sem instalar ou baixar arquivos")
    argumentos = parser.parse_args()
    try:
        verificar() if argumentos.verificar else instalar()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Não foi possível preparar o ambiente: {exc}", file=sys.stderr)
        print("Confira o Python instalado e a conexão com a internet. "
              "Se .venv veio de outro computador, remova essa pasta e execute o instalador novamente.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
