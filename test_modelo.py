import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from modelo import Config, construir_rede, separar, separar_tres, treinar


class TestModelo(unittest.TestCase):
    def test_divisao_exata_disjunta_repetivel_com_classes(self):
        y = np.tile(np.arange(10), 100)
        c = Config(treino=650, teste=150)
        treino, teste = separar(y, c)
        self.assertEqual((len(treino), len(teste)), (650, 150))
        self.assertEqual(len(set(treino) & set(teste)), 0)
        self.assertEqual(set(y[treino]), set(range(10)))
        np.testing.assert_array_equal(treino, separar(y, c)[0])
        for arquitetura in ('CNN', 'LeNet', 'RBF', 'SOM'):
            np.testing.assert_array_equal(teste, separar(y, Config(arquitetura=arquitetura, treino=650, teste=150))[1])

    def test_rejeita_configuracoes_invalidas(self):
        for campos in [dict(treino=69000, teste=2000), dict(taxa=float('nan')),
                       dict(treino=20.5), dict(epocas=0), dict(arquitetura='RNN'),
                       dict(teste=-1), dict(taxa=True), dict(som_linhas=1),
                       dict(som_sigma=float('nan')), dict(rbf_sigma=0),
                       dict(arquitetura='RBF', treino=10, neuronios=11),
                       dict(arquitetura='LeNet', treino=56000, teste=14000, validacao=10),
                       dict(arquitetura='LeNet', validacao=5),
                       dict(arquitetura='SOM', validacao=10), dict(perfil_cnn='desconhecido')]:
            with self.subTest(campos=campos), self.assertRaises(ValueError):
                Config(**campos).validar()

    def test_quatro_redes_treinam_e_modelo_salvo_reproduz_previsao(self):
        rng = np.random.default_rng(2)
        y = np.tile(np.arange(10), 10)
        X = rng.random((100, 784), dtype=np.float32)
        for arquitetura in ['MLP', 'CNN', 'LeNet', 'RBF', 'SOM']:
            with self.subTest(arquitetura=arquitetura), tempfile.TemporaryDirectory() as pasta:
                eventos = []
                c = Config(arquitetura=arquitetura, treino=60, teste=20, epocas=2, neuronios=16)
                r = treinar(c, lambda **kw: eventos.append(kw), dados=(X, y), destino=pasta)
                self.assertEqual(len(r['perdas']), 2)
                self.assertEqual(sum(map(sum, r['matriz'])), 20)
                self.assertEqual(r['nao_utilizadas'], 20)
                self.assertEqual([e['epoca'] for e in eventos if 'epoca' in e], [1, 2])
                pacote = torch.load(Path(r['pasta']) / 'modelo.pt', weights_only=True)
                rede = construir_rede(Config(**pacote['config']))
                rede.load_state_dict(pacote['pesos'])
                rede.eval()
                with np.load(Path(r['pasta']) / 'divisao.npz') as arquivo:
                    indices = arquivo['teste'].copy()
                with torch.inference_mode():
                    previstos = rede(torch.from_numpy(X[indices[:12]].reshape(-1,1,28,28))).argmax(1).tolist()
                self.assertEqual(previstos, [e['previsto'] for e in r['exemplos']])

    def test_lenet_formato_gradientes_e_quantidade_de_parametros(self):
        torch.set_num_threads(2)
        rede = construir_rede(Config(arquitetura='LeNet'))
        self.assertEqual(sum(p.numel() for p in rede.parameters()), 61706)
        logits = rede(torch.rand(4, 1, 28, 28))
        self.assertEqual(tuple(logits.shape), (4, 10))
        torch.nn.functional.cross_entropy(logits, torch.arange(4)).backward()
        for p in rede.parameters():
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(p.grad.abs().sum().item(), 0)

    def test_lenet_validacao_separada_e_teste_nao_altera_modelo(self):
        rng = np.random.default_rng(4)
        X = rng.random((200, 784), dtype=np.float32)
        y = np.tile(np.arange(10), 20)
        c = Config(arquitetura='LeNet', treino=120, validacao=30, teste=50, epocas=3)
        treino, validacao, teste = separar_tres(y, c)
        self.assertEqual(tuple(map(len, (treino, validacao, teste))), (120, 30, 50))
        self.assertEqual(len(set(treino) | set(validacao) | set(teste)), 200)
        for indices in (treino, validacao, teste):
            self.assertEqual(set(y[indices]), set(range(10)))
        np.testing.assert_array_equal(teste, separar(y, Config(treino=150, teste=50))[1])
        modificado = X.copy()
        modificado[teste] = 1 - modificado[teste]
        with tempfile.TemporaryDirectory() as pasta:
            a = treinar(c, dados=(X, y), destino=pasta)
            b = treinar(c, dados=(modificado, y), destino=pasta)
            self.assertEqual(a['melhor_epoca'], int(np.argmin(a['perdas_validacao'])) + 1)
            self.assertEqual(a['melhor_epoca'], b['melhor_epoca'])
            self.assertEqual(len(a['acuracias_validacao']), c.epocas)
            pa = torch.load(Path(a['pasta']) / 'modelo.pt', weights_only=True)
            pb = torch.load(Path(b['pasta']) / 'modelo.pt', weights_only=True)
            for chave in pa['pesos']:
                torch.testing.assert_close(pa['pesos'][chave], pb['pesos'][chave], rtol=0, atol=0)
            rede = construir_rede(Config(**pa['config']))
            rede.load_state_dict(pa['pesos'])
            rede.eval()
            with torch.inference_mode():
                previstos = rede(torch.from_numpy(X[teste].reshape(-1, 1, 28, 28))).argmax(1).numpy()
            with np.load(Path(a['pasta']) / 'previsoes_teste.npz') as salvo:
                np.testing.assert_array_equal(salvo['previstos'], previstos)

    def test_rbf_gaussiana_e_centros_fixos(self):
        from modelos_classicos import RedeRBF
        rede = RedeRBF(2, 2)
        with torch.no_grad():
            rede.centros[1].fill_(1)
            rede.saida.weight.zero_()
            rede.saida.bias.zero_()
            rede.saida.weight[0, 0] = 1
            rede.saida.weight[1, 1] = 1
        x = torch.zeros(1, 784)
        saidas = rede(x)
        self.assertEqual(saidas[0, 0].item(), 1)
        self.assertAlmostEqual(saidas[0, 1].item(), torch.exp(torch.tensor(-784 / 8)).item())
        saidas.sum().backward()
        self.assertIsNone(rede.centros.grad)

    def test_som_aprende_sem_rotulos_e_rotula_unidades_vazias(self):
        from modelos_classicos import MapaSOM, ajustar_som
        c = Config(arquitetura='SOM', treino=20, teste=10, epocas=3,
                   som_linhas=3, som_colunas=3)
        X = torch.cat([torch.zeros(10, 784), torch.ones(10, 784)])
        redes = []
        for digitos in ([2]*10+[7]*10, [3]*10+[8]*10):
            rede = MapaSOM(3, 3)
            ajustar_som(rede, X, torch.tensor(digitos), c, lambda *args: None)
            redes.append(rede)
            self.assertEqual(rede.contagens.sum().item(), 20)
            self.assertTrue(set(rede.rotulos.tolist()) <= set(digitos))
            self.assertTrue((rede.contagens.sum(1) == 0).any())
        torch.testing.assert_close(redes[0].prototipos, redes[1].prototipos)
        self.assertEqual(redes[0](X[[0, 19]]).argmax(1).tolist(), [2, 7])

    def test_dados_de_teste_nao_alteram_rbf_ou_som(self):
        rng = np.random.default_rng(4)
        X = rng.random((100, 784), dtype=np.float32)
        y = np.tile(np.arange(10), 10)
        for arquitetura in ['RBF', 'SOM']:
            c = Config(arquitetura=arquitetura, treino=60, teste=20, epocas=2, neuronios=8,
                       som_linhas=3, som_colunas=3)
            _, teste = separar(y, c)
            modificado = X.copy()
            modificado[teste] = 1 - modificado[teste]
            with self.subTest(arquitetura=arquitetura), tempfile.TemporaryDirectory() as pasta:
                a = treinar(c, dados=(X,y), destino=pasta)
                b = treinar(c, dados=(modificado,y), destino=pasta)
                pa = torch.load(Path(a['pasta'])/'modelo.pt', weights_only=True)['pesos']
                pb = torch.load(Path(b['pasta'])/'modelo.pt', weights_only=True)['pesos']
                for chave in pa:
                    torch.testing.assert_close(pa[chave], pb[chave], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
