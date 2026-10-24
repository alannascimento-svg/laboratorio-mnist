import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from arquiteturas import (NOVAS, Inception, BlocoResidual, ResidualInvertido,
                         descrever)
from modelo import Config, construir_rede, separar_tres, treinar


class TestArquiteturas(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_familias_preservam_blocos_e_profundidade(self):
        for nome in NOVAS:
            with self.subTest(nome=nome):
                rede = construir_rede(Config(arquitetura=nome)).rede
                if nome == 'AlexNet':
                    self.assertEqual(sum(isinstance(m, nn.Conv2d) for m in rede.modules()), 5)
                    self.assertEqual(sum(isinstance(m, nn.Linear) for m in rede.modules()), 3)
                if nome == 'VGG19':
                    self.assertEqual(sum(isinstance(m, nn.Conv2d) for m in rede.modules()), 16)
                    self.assertEqual(sum(isinstance(m, nn.Linear) for m in rede.modules()), 3)
                if nome == 'GoogLeNet':
                    self.assertEqual(sum(isinstance(m, Inception) for m in rede.modules()), 9)
                if nome == 'ResNet18':
                    self.assertEqual(sum(isinstance(m, BlocoResidual) for m in rede.modules()), 8)
                if nome == 'MobileNetV2':
                    self.assertEqual(sum(isinstance(m, ResidualInvertido) for m in rede.modules()), 17)
                    self.assertTrue(any(isinstance(m, nn.Conv2d) and m.groups > 1 for m in rede.modules()))

    def test_perfis_classicos_e_compactos_recebem_imagens_mnist(self):
        for nome in NOVAS:
            contagens = []
            for perfil in ('compacto', 'classico'):
                with self.subTest(nome=nome, perfil=perfil):
                    rede = construir_rede(Config(arquitetura=nome, perfil_cnn=perfil))
                    rede.eval()
                    with torch.inference_mode():
                        logits = rede(torch.rand(2, 1, 28, 28))
                    self.assertEqual(tuple(logits.shape), (2, 10))
                    self.assertTrue(torch.isfinite(logits).all())
                    contagens.append(sum(p.numel() for p in rede.parameters()))
                    del rede
            self.assertGreater(contagens[1], contagens[0])

    def test_lote_unitario_treina_sem_falha_em_batchnorm(self):
        for nome in NOVAS:
            with self.subTest(nome=nome):
                rede = construir_rede(Config(arquitetura=nome))
                logits = rede(torch.rand(1, 1, 28, 28))
                nn.functional.cross_entropy(logits, torch.tensor([3])).backward()
                gradientes = [p.grad for p in rede.parameters() if p.grad is not None]
                self.assertTrue(all(torch.isfinite(g).all() for g in gradientes))
                self.assertTrue(any(g.abs().sum() > 0 for g in gradientes))

    def test_treinamento_validacao_recarga_e_particoes_identicas(self):
        rng = np.random.default_rng(4)
        X = rng.random((100, 784), dtype=np.float32)
        y = np.tile(np.arange(10), 10)
        referencia = separar_tres(y, Config(arquitetura='LeNet', treino=60, validacao=20, teste=20))
        for nome in NOVAS:
            with self.subTest(nome=nome), tempfile.TemporaryDirectory() as pasta:
                c = Config(arquitetura=nome, treino=60, validacao=20, teste=20, epocas=2, lote=16)
                r = treinar(c, dados=(X, y), destino=pasta)
                self.assertEqual(len(r['perdas']), 2)
                self.assertEqual(len(r['perdas_validacao']), 2)
                self.assertEqual(r['melhor_epoca'], int(np.argmin(r['perdas_validacao'])) + 1)
                self.assertEqual(sum(map(sum, r['matriz'])), 20)
                self.assertEqual(r['arquitetura_detalhes'], descrever(nome, 'compacto'))
                with np.load(Path(r['pasta']) / 'divisao.npz') as arquivo:
                    indices = {chave: arquivo[chave].copy() for chave in arquivo.files}
                for chave, esperado in zip(('treino', 'validacao', 'teste'), referencia):
                    np.testing.assert_array_equal(indices[chave], esperado)
                pacote = torch.load(Path(r['pasta']) / 'modelo.pt', weights_only=True)
                rede = construir_rede(Config(**pacote['config']))
                rede.load_state_dict(pacote['pesos'])
                rede.eval()
                with torch.inference_mode():
                    previstos = rede(torch.from_numpy(X[indices['teste']].reshape(-1, 1, 28, 28))).argmax(1).numpy()
                with np.load(Path(r['pasta']) / 'previsoes_teste.npz') as salvo:
                    np.testing.assert_array_equal(previstos, salvo['previstos'])
                self.assertTrue((Path(r['pasta']) / 'codigo_fonte' / 'arquiteturas.py').exists())

    def test_residuos_somam_atalho_e_inception_concatena_ramos(self):
        bloco = BlocoResidual(4, 4).eval()
        for p in bloco.conv2.parameters():
            nn.init.zeros_(p)
        x = torch.rand(2, 4, 8, 8)
        torch.testing.assert_close(bloco(x), x)
        invertido = ResidualInvertido(8, 8, 1, 6).eval()
        for p in invertido.block[-1].parameters():
            nn.init.zeros_(p)
        x = torch.rand(2, 8, 8, 8)
        torch.testing.assert_close(invertido(x), x)
        inception = Inception(8, (4, 4, 8, 4, 4, 4)).eval()
        self.assertEqual(tuple(inception(x).shape), (2, 20, 8, 8))


if __name__ == '__main__':
    unittest.main()
