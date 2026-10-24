#!/usr/bin/env bash
# Também funciona no macOS; execute com bash instalar_linux.sh.
set -euo pipefail
cd -- "$(dirname -- "$0")"
for candidato in python3.12 python3.13 python3.11 python3; do
    if command -v "$candidato" >/dev/null 2>&1 && "$candidato" -c 'import sys; sys.exit(sys.version_info[:2] not in ((3,11),(3,12),(3,13)))' >/dev/null 2>&1; then
        exec "$candidato" instalar.py "$@"
    fi
done
echo 'Instale Python 3.12 (ou 3.11/3.13) e execute este instalador novamente.' >&2
exit 1
