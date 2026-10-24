"""Painel MNIST local, com treinamento isolado e interrupção imediata."""
import argparse
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

from modelo import BASE, Config

lock = threading.Lock()
processo = None
parada_solicitada = False
PADRAO = Config(arquitetura='LeNet', treino=2000, validacao=500, teste=500,
                epocas=3, lote=64, perfil_cnn='compacto')


def inicial():
    return {'ativo': False, 'fase': 'Pronto para treinar', 'epoca': 0,
            'perdas': [], 'perdas_validacao': [], 'acuracias_validacao': [],
            'config': asdict(PADRAO), 'resultado': None, 'erro': None}


estado = inicial()


def restaurar():
    eventos = sorted(list((BASE / 'resultados').glob('*/resultado.json')) +
                     list((BASE / 'resultados').glob('*/interrompido.json')), reverse=True)
    for caminho in eventos:
        try:
            salvo = json.loads(caminho.read_text(encoding='utf-8'))
            c = Config(**salvo['config'])
            c.validar()
            if caminho.name == 'interrompido.json':
                estado.update(salvo, ativo=False, resultado=None, erro=None)
            else:
                estado.update(config=asdict(c), fase='Treinamento concluído',
                              epoca=c.epocas, resultado=salvo, perdas=salvo['perdas'],
                              perdas_validacao=salvo.get('perdas_validacao', []),
                              acuracias_validacao=salvo.get('acuracias_validacao', []))
            return
        except (OSError, ValueError, TypeError, KeyError):
            continue


def acompanhar(proc):
    global processo
    resultado, erro = None, None
    for linha in proc.stdout:
        try:
            evento = json.loads(linha)
        except ValueError:
            continue  # Avisos das bibliotecas não fazem parte do protocolo JSON.
        with lock:
            if processo is not proc:
                continue
            if evento.get('tipo') == 'progresso' and not parada_solicitada:
                estado.update(evento['campos'])
            elif evento.get('tipo') == 'resultado':
                resultado = evento['resultado']
            elif evento.get('tipo') == 'erro':
                erro = evento['mensagem']
    codigo = proc.wait()
    proc.stdout.close()
    with lock:
        if processo is not proc:
            return
        if parada_solicitada:
            estado.update(ativo=False, fase='Treinamento interrompido pelo usuário',
                          resultado=None, erro=None)
            pasta = estado.get('pasta_execucao')
            if pasta:
                try:
                    Path(pasta, 'interrompido.json').write_text(
                        json.dumps(estado, ensure_ascii=False, indent=2), encoding='utf-8')
                except OSError:
                    pass
        elif resultado is not None and codigo == 0:
            estado.update(ativo=False, fase='Treinamento concluído', resultado=resultado)
        else:
            estado.update(ativo=False, fase='Não foi possível concluir o treinamento',
                          erro=erro or 'O processo de treinamento terminou antes de concluir.')
        processo = None


def iniciar(config):
    global processo, parada_solicitada
    with lock:
        if estado['ativo']:
            return False
        ambiente = os.environ.copy()
        ambiente.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
        proc = subprocess.Popen([sys.executable, str(BASE / 'executar_treino.py')],
                                cwd=BASE, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                                env=ambiente)
        try:
            proc.stdin.write(json.dumps(asdict(config)))
            proc.stdin.close()
        except (OSError, BrokenPipeError):
            proc.kill()
            proc.wait()
            proc.stdout.close()
            raise
        processo, parada_solicitada = proc, False
        estado.clear()
        estado.update(inicial(), ativo=True, fase='Preparando treinamento', config=asdict(config))
        threading.Thread(target=acompanhar, args=(proc,), daemon=True).start()
        return True


def parar():
    global parada_solicitada
    with lock:
        if not estado['ativo'] or processo is None:
            return False
        parada_solicitada = True
        estado['fase'] = 'Interrompendo treinamento…'
        if processo.poll() is None:
            processo.terminate()
        return True


class Handler(BaseHTTPRequestHandler):
    def responder(self, valor, status=200):
        dados = json.dumps(valor, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(dados)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(dados)

    def do_GET(self):
        if self.path == '/api/historico':
            itens = []
            for caminho in sorted((BASE / 'resultados').glob('*/resultado.json'), reverse=True):
                try:
                    r = json.loads(caminho.read_text(encoding='utf-8'))
                    itens.append({'id': caminho.parent.name, 'config': r['config'],
                                  'acuracia': r['acuracia'], 'segundos': r.get('segundos'),
                                  'parametros': r.get('parametros'), 'melhor_epoca': r.get('melhor_epoca')})
                except (ValueError, OSError, KeyError):
                    continue
            return self.responder(itens)
        if self.path == '/api/status':
            with lock:
                resposta = dict(estado)
            return self.responder(resposta)
        if self.path == '/':
            pagina = (BASE / 'interface.html').read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(pagina)))
            self.end_headers()
            return self.wfile.write(pagina)
        if self.path == '/favicon.ico':
            self.send_response(204)
            return self.end_headers()
        self.responder({'erro': 'Página não encontrada'}, 404)

    def do_POST(self):
        if self.path not in ('/api/treinar', '/api/parar'):
            return self.responder({'erro': 'Página não encontrada'}, 404)
        origem = self.headers.get('Origin')
        if origem and origem != 'http://' + self.headers.get('Host', ''):
            return self.responder({'erro': 'Origem não permitida'}, 403)
        if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
            return self.responder({'erro': 'Envie dados JSON'}, 415)
        try:
            tamanho = int(self.headers.get('Content-Length', 0))
            if not 0 < tamanho <= 4096:
                raise ValueError('Configuração inválida')
            dados = json.loads(self.rfile.read(tamanho))
            if not isinstance(dados, dict):
                raise ValueError('A configuração deve ser um objeto JSON')
            if self.path == '/api/parar':
                if parar():
                    return self.responder({'ok': True}, 202)
                return self.responder({'erro': 'Não há treinamento em andamento'}, 409)
            config = Config(**dados)
            config.validar()
        except (TypeError, ValueError) as exc:
            return self.responder({'erro': str(exc)}, 400)
        try:
            if not iniciar(config):
                return self.responder({'erro': 'Já existe um treinamento em andamento'}, 409)
        except OSError:
            return self.responder({'erro': 'Não foi possível iniciar o processo de treinamento'}, 500)
        self.responder({'ok': True}, 202)

    def log_message(self, formato, *args):
        if not args or '/api/status' not in str(args[0]):
            super().log_message(formato, *args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8503)
    args = parser.parse_args()
    restaurar()
    try:
        servidor = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    except OSError as exc:
        parser.exit(1, f'Não foi possível abrir a porta {args.port}: {exc}\nUse --port 8504.\n')
    print(f'MNIST: http://127.0.0.1:{args.port}', flush=True)
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        parar()
        servidor.server_close()


if __name__ == '__main__':
    main()
