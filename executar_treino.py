"""Processo isolado: recebe configuração JSON e emite progresso em JSON Lines."""
import json
import sys


def emitir(**evento):
    print(json.dumps(evento, ensure_ascii=False), flush=True)


def main():
    try:
        from modelo import Config, treinar
        config = Config(**json.load(sys.stdin))
        resultado = treinar(config, informar=lambda **kw: emitir(tipo='progresso', campos=kw))
        emitir(tipo='resultado', resultado=resultado)
    except Exception as exc:
        emitir(tipo='erro', mensagem=str(exc))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
