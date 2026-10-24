"""Testes HTTP locais com um processo de treino simulado, sem baixar dados."""
from dataclasses import asdict
import http.client
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import app


# O processo é real; somente o trabalho científico é substituído por um protocolo
# JSON Lines mínimo. Assim a parada é verificada sem treinar ou acessar a rede.
WORKER = r'''
import json
import os
from pathlib import Path
import sys
import time

config = json.load(sys.stdin)
pasta = Path('resultados') / ('r_simulado_%d' % os.getpid())
pasta.mkdir(parents=True)
(pasta / 'config.json').write_text(json.dumps(config), encoding='utf-8')
print('Aviso fora do protocolo JSON', flush=True)
print(json.dumps({'tipo': 'progresso', 'campos': {
    'fase': 'Processo simulado aguardando parada',
    'epoca': 1, 'perdas': [2.5], 'pasta_execucao': str(pasta.resolve())
}}), flush=True)
if config['semente'] == 100:
    resultado = {'config': config, 'perdas': [2.5], 'acuracia': 0.5,
                 'segundos': 0.01, 'parametros': 1, 'melhor_epoca': 1,
                 'pasta': str(pasta.resolve())}
    (pasta / 'resultado.json').write_text(json.dumps(resultado), encoding='utf-8')
    print(json.dumps({'tipo': 'resultado', 'resultado': resultado}), flush=True)
elif config['semente'] == 101:
    print(json.dumps({'tipo': 'erro', 'mensagem': 'Falha simulada'}), flush=True)
    raise SystemExit(1)
else:
    while True:
        time.sleep(0.05)
'''


class HandlerSilencioso(app.Handler):
    def log_message(self, formato, *args):
        pass


class TestAppHTTP(unittest.TestCase):
    def setUp(self):
        temporario = tempfile.TemporaryDirectory()
        self.addCleanup(temporario.cleanup)
        self.base = Path(temporario.name)
        self.pagina = '<!doctype html><html lang="pt-BR">Painel de teste</html>'
        (self.base / 'interface.html').write_text(self.pagina, encoding='utf-8')
        (self.base / 'executar_treino.py').write_text(WORKER, encoding='utf-8')
        self.processos = []
        for nome, valor in [('BASE', self.base), ('estado', app.inicial()),
                            ('processo', None), ('parada_solicitada', False)]:
            remendo = patch.object(app, nome, valor)
            remendo.start()
            self.addCleanup(remendo.stop)
        self.servidor = ThreadingHTTPServer(('127.0.0.1', 0), HandlerSilencioso)
        self.porta = self.servidor.server_address[1]
        self.thread = threading.Thread(
            target=self.servidor.serve_forever, kwargs={'poll_interval': 0.02}, daemon=True)
        self.thread.start()
        # Registrado por último: encerra os processos antes de desfazer os patches
        # e remover os arquivos temporários usados pelo leitor de progresso.
        self.addCleanup(self.fechar)

    def fechar(self):
        app.parar()
        for proc in self.processos:
            if proc.poll() is None:
                proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)
        limite = time.monotonic() + 3
        while time.monotonic() < limite:
            with app.lock:
                pendente = app.processo
            if pendente is None:
                break
            time.sleep(0.01)
        self.servidor.shutdown()
        self.servidor.server_close()
        self.thread.join(timeout=3)
        self.assertFalse(self.thread.is_alive(), 'O servidor de teste ficou ativo.')
        self.assertIsNone(app.processo, 'O leitor de progresso não encerrou.')
        self.assertTrue(all(proc.poll() is not None for proc in self.processos),
                        'Um processo simulado ficou órfão.')

    def requisitar(self, metodo, caminho, dados=None, corpo=None, cabecalhos=None):
        headers = dict(cabecalhos or {})
        if metodo == 'POST':
            headers.setdefault('Content-Type', 'application/json')
            if corpo is None:
                corpo = json.dumps({} if dados is None else dados).encode('utf-8')
        conexao = http.client.HTTPConnection('127.0.0.1', self.porta, timeout=5)
        try:
            conexao.request(metodo, caminho, body=corpo, headers=headers)
            resposta = conexao.getresponse()
            conteudo = resposta.read()
            tipo = resposta.getheader('Content-Type', '')
            valor = json.loads(conteudo) if tipo.startswith('application/json') else conteudo
            return resposta.status, valor, dict(resposta.getheaders())
        finally:
            conexao.close()

    def esperar_estado(self, condicao):
        limite = time.monotonic() + 5
        ultimo = None
        while time.monotonic() < limite:
            codigo, ultimo, _ = self.requisitar('GET', '/api/status')
            self.assertEqual(codigo, 200)
            if condicao(ultimo):
                return ultimo
            time.sleep(0.02)
        self.fail(f'O painel não atingiu o estado esperado: {ultimo}')

    def iniciar(self, semente=42):
        config = asdict(app.PADRAO)
        config['semente'] = semente
        codigo, resposta, _ = self.requisitar('POST', '/api/treinar', config)
        self.assertEqual((codigo, resposta), (202, {'ok': True}))
        # Captura o Popen para provar a existência do filho e assegurar limpeza,
        # mesmo se uma asserção posterior falhar antes de pedir a parada.
        with app.lock:
            proc = app.processo
        if proc is not None:
            self.processos.append(proc)
        return config, proc

    def test_paginas_status_e_historico_nao_iniciam_treino(self):
        codigo, pagina, headers = self.requisitar('GET', '/')
        self.assertEqual(codigo, 200)
        self.assertEqual(pagina.decode('utf-8'), self.pagina)
        self.assertEqual(headers['Content-Type'], 'text/html; charset=utf-8')
        codigo, estado, headers = self.requisitar('GET', '/api/status')
        self.assertEqual(codigo, 200)
        self.assertFalse(estado['ativo'])
        self.assertIsNone(estado['resultado'])
        self.assertEqual(headers['Cache-Control'], 'no-store')
        self.assertEqual(self.requisitar('GET', '/api/historico')[:2], (200, []))
        self.assertEqual(self.requisitar('GET', '/favicon.ico')[0], 204)
        self.assertEqual(self.requisitar('GET', '/inexistente')[0], 404)

        pasta = self.base / 'resultados' / 'r_historico'
        pasta.mkdir(parents=True)
        salvo = {'config': asdict(app.PADRAO), 'acuracia': 0.75,
                 'segundos': 1.5, 'parametros': 61706, 'melhor_epoca': 2}
        (pasta / 'resultado.json').write_text(json.dumps(salvo), encoding='utf-8')
        invalido = self.base / 'resultados' / 'r_invalido'
        invalido.mkdir()
        (invalido / 'resultado.json').write_text('{', encoding='utf-8')
        codigo, historico, _ = self.requisitar('GET', '/api/historico')
        self.assertEqual(codigo, 200)
        self.assertEqual(len(historico), 1)
        self.assertEqual(historico[0], dict(salvo, id='r_historico'))
        self.assertIsNone(app.processo)

    def test_rejeita_origem_conteudo_e_configuracoes_invalidas(self):
        config = asdict(app.PADRAO)
        for rota in ('/api/treinar', '/api/parar'):
            with self.subTest(rota=rota):
                self.assertEqual(self.requisitar('POST', rota, config,
                    cabecalhos={'Origin': 'https://origem-invalida.example'})[0], 403)
        self.assertEqual(self.requisitar('POST', '/api/treinar', config,
            cabecalhos={'Content-Type': 'text/plain'})[0], 415)
        self.assertEqual(self.requisitar('POST', '/inexistente', config)[0], 404)
        casos = [([], None, {}), (None, b'{', {}),
                 ({'extra': 'x' * 5000}, None, {}),
                 (config, None, {'Content-Length': '0'}),
                 (config, None, {'Content-Length': '-1'}),
                 (config, None, {'Content-Length': 'invalido'}),
                 (dict(config, arquitetura='Inexistente'), None, {}),
                 (dict(config, perfil_cnn='inexistente'), None, {}),
                 (dict(config, epocas=0), None, {}),
                 (dict(config, campo_desconhecido=True), None, {})]
        for i, (dados, corpo, cabecalhos) in enumerate(casos):
            with self.subTest(caso=i):
                codigo, resposta, _ = self.requisitar('POST', '/api/treinar', dados,
                                                     corpo=corpo, cabecalhos=cabecalhos)
                self.assertEqual(codigo, 400)
                self.assertIn('erro', resposta)
        self.assertEqual(self.requisitar('POST', '/api/parar', [])[0], 400)
        self.assertEqual(self.requisitar('POST', '/api/parar', {})[0], 409)
        self.assertIsNone(app.processo)
        self.assertFalse(app.estado['ativo'])

    def test_treino_exclusivo_parada_real_e_novo_treino(self):
        config, primeiro = self.iniciar()
        estado = self.esperar_estado(lambda s: s['ativo'] and s['epoca'] == 1)
        self.assertIsNotNone(primeiro)
        self.assertIsNone(primeiro.poll())
        self.assertEqual(estado['perdas'], [2.5])
        self.assertEqual(estado['config'], config)
        self.assertEqual(self.requisitar('POST', '/api/treinar', config)[0], 409)
        # Uma origem externa também não pode interromper um treino legítimo.
        self.assertEqual(self.requisitar('POST', '/api/parar', {},
            cabecalhos={'Origin': 'https://origem-invalida.example'})[0], 403)
        self.assertIsNone(primeiro.poll())
        origem_local = f'http://127.0.0.1:{self.porta}'
        codigo, resposta, _ = self.requisitar('POST', '/api/parar', {},
                                             cabecalhos={'Origin': origem_local})
        self.assertEqual((codigo, resposta), (202, {'ok': True}))
        parado = self.esperar_estado(lambda s: not s['ativo'])
        self.assertEqual(parado['fase'], 'Treinamento interrompido pelo usuário')
        self.assertIsNone(parado['resultado'])
        self.assertIsNone(parado['erro'])
        self.assertIsNotNone(primeiro.poll())
        self.assertTrue(primeiro.stdout.closed)
        arquivo = Path(estado['pasta_execucao']) / 'interrompido.json'
        self.assertTrue(arquivo.is_file())
        self.assertFalse(json.loads(arquivo.read_text(encoding='utf-8'))['ativo'])
        self.assertEqual(self.requisitar('POST', '/api/parar', {})[0], 409)

        _, segundo = self.iniciar(semente=43)
        self.esperar_estado(lambda s: s['ativo'] and s['epoca'] == 1)
        self.assertIsNotNone(segundo)
        self.assertNotEqual(primeiro.pid, segundo.pid)
        self.assertEqual(self.requisitar('POST', '/api/parar', {})[0], 202)
        self.esperar_estado(lambda s: not s['ativo'])
        self.assertIsNotNone(segundo.poll())

    def test_resultado_concluido_e_lido_no_historico(self):
        self.iniciar(semente=100)
        estado = self.esperar_estado(lambda s: not s['ativo'])
        self.assertEqual(estado['fase'], 'Treinamento concluído')
        self.assertIsNone(estado['erro'])
        self.assertEqual(estado['resultado']['acuracia'], 0.5)
        self.assertIsNone(app.processo)
        codigo, historico, _ = self.requisitar('GET', '/api/historico')
        self.assertEqual(codigo, 200)
        self.assertEqual(len(historico), 1)
        self.assertEqual(historico[0]['acuracia'], 0.5)

    def test_falha_do_processo_libera_painel(self):
        self.iniciar(semente=101)
        estado = self.esperar_estado(lambda s: not s['ativo'])
        self.assertEqual(estado['erro'], 'Falha simulada')
        self.assertIsNone(estado['resultado'])
        self.assertIsNone(app.processo)
        self.assertEqual(self.requisitar('POST', '/api/parar', {})[0], 409)
        self.iniciar(semente=100)
        concluido = self.esperar_estado(lambda s: not s['ativo'])
        self.assertEqual(concluido['fase'], 'Treinamento concluído')
        self.assertIsNone(concluido['erro'])

    def test_restauracao_considera_ultima_execucao_valida(self):
        resultado = {'config': asdict(app.PADRAO), 'perdas': [2.5],
                     'acuracia': 0.5, 'pasta': 'resultado-anterior'}
        concluida = self.base / 'resultados' / 'r001'
        concluida.mkdir(parents=True)
        (concluida / 'resultado.json').write_text(json.dumps(resultado), encoding='utf-8')
        app.restaurar()
        self.assertFalse(app.estado['ativo'])
        self.assertEqual(app.estado['resultado'], resultado)
        self.assertEqual(app.estado['fase'], 'Treinamento concluído')

        interrupcao = dict(app.inicial(), fase='Treinamento interrompido pelo usuário',
                           epoca=1, perdas=[2.5], ativo=True)
        interrompida = self.base / 'resultados' / 'r002'
        interrompida.mkdir()
        (interrompida / 'interrompido.json').write_text(json.dumps(interrupcao), encoding='utf-8')
        corrompida = self.base / 'resultados' / 'r003'
        corrompida.mkdir()
        (corrompida / 'resultado.json').write_text('{', encoding='utf-8')
        app.restaurar()
        self.assertFalse(app.estado['ativo'])
        self.assertIsNone(app.estado['resultado'])
        self.assertEqual(app.estado['fase'], 'Treinamento interrompido pelo usuário')
        self.assertEqual(app.estado['epoca'], 1)
        self.assertIsNone(app.processo)


if __name__ == '__main__':
    unittest.main()
