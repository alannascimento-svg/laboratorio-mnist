#!/usr/bin/env bash
# Também funciona no macOS; aceita, por exemplo, --port 8504.
set -euo pipefail
cd -- "$(dirname -- "$0")"
if [ ! -x .venv/bin/python ]; then
    echo 'Execute primeiro: bash instalar_linux.sh' >&2
    exit 1
fi
if ! .venv/bin/python instalar.py --verificar; then
    echo 'Confira o ambiente executando: bash instalar_linux.sh' >&2
    exit 1
fi
exec .venv/bin/python app.py "$@"
