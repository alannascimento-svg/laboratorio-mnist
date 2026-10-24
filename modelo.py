"""Treinamento MNIST com divisão configurável e teste separado."""
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import json
import time
import shutil

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

BASE = Path(__file__).resolve().parent
NOVAS_CNN = ('AlexNet', 'VGG19', 'GoogLeNet', 'ResNet18', 'MobileNetV2')
ARQUITETURAS = ('MLP', 'CNN', 'LeNet', *NOVAS_CNN, 'RBF', 'SOM')


@dataclass
class Config:
    arquitetura: str = 'MLP'
    treino: int = 56000
    teste: int = 14000
    epocas: int = 20
    neuronios: int = 128
    lote: int = 128
    taxa: float = 0.001
    semente: int = 42
    rbf_sigma: float = 5.0
    som_linhas: int = 10
    som_colunas: int = 10
    som_sigma: float = 3.0
    validacao: int = 0
    perfil_cnn: str = 'compacto'

    def validar(self, total=70000):
        if self.arquitetura not in ARQUITETURAS:
            raise ValueError('Selecione uma das arquiteturas disponíveis no painel.')
        if self.perfil_cnn not in ('compacto', 'classico'):
            raise ValueError('Selecione o perfil CNN compacto ou clássico.')
        limites = {'treino': (10, total), 'teste': (10, total),
                   'epocas': (1, 1000), 'neuronios': (1, 2048),
                   'lote': (1, 4096), 'semente': (0, 2**32 - 1),
                   'som_linhas': (2, 30), 'som_colunas': (2, 30),
                   'validacao': (0, total)}
        for campo, (minimo, maximo) in limites.items():
            valor = getattr(self, campo)
            if type(valor) is not int or not minimo <= valor <= maximo:
                raise ValueError(f'{campo}: informe um inteiro entre {minimo} e {maximo}.')
        if self.treino + self.validacao + self.teste > total:
            raise ValueError(f'Treino + validação + teste não pode ultrapassar {total:,} imagens.')
        if self.validacao and (self.validacao < 10 or self.arquitetura == 'SOM'):
            raise ValueError('Validação requer modelo supervisionado e pelo menos 10 imagens.')
        if isinstance(self.taxa, bool) or not isinstance(self.taxa, (int, float)) or not 0 < self.taxa <= 1:
            raise ValueError('A taxa de aprendizado deve ser maior que 0 e até 1.')
        for campo, minimo, maximo in [('rbf_sigma', 0.1, 100), ('som_sigma', 0.5, 30)]:
            valor = getattr(self, campo)
            if isinstance(valor, bool) or not isinstance(valor, (int, float)) or not minimo <= valor <= maximo:
                raise ValueError(f'{campo}: informe um valor entre {minimo} e {maximo}.')
        if self.arquitetura == 'RBF' and self.neuronios > self.treino:
            raise ValueError('A quantidade de centros RBF deve ser no máximo a quantidade de imagens de treino.')


def carregar():
    X, y = fetch_openml('mnist_784', version=1, as_frame=False,
                        return_X_y=True, data_home=str(BASE / 'dados'))
    if X.shape != (70000, 784):
        raise ValueError(f'Formato inesperado do MNIST: {X.shape}.')
    return X.astype(np.float32) / 255.0, y.astype(np.int64)


def separar(y, config):
    config.validar(len(y))
    # A divisão opera nos índices: nenhum exemplo pertence aos dois conjuntos.
    return train_test_split(np.arange(len(y)), train_size=config.treino + config.validacao,
                            test_size=config.teste, random_state=config.semente,
                            stratify=y)


def separar_tres(y, config):
    candidatos, teste = separar(y, config)
    if not config.validacao:
        return candidatos, np.array([], dtype=np.int64), teste
    treino, validacao = train_test_split(
        candidatos, train_size=config.treino, test_size=config.validacao,
        random_state=config.semente, stratify=y[candidatos])
    return treino, validacao, teste


def construir_rede(config):
    from torch import nn
    from modelos_classicos import RedeRBF, MapaSOM
    if config.arquitetura in NOVAS_CNN:
        from arquiteturas import construir
        return construir(config.arquitetura, config.perfil_cnn)
    if config.arquitetura == 'RBF':
        return RedeRBF(config.neuronios, config.rbf_sigma)
    if config.arquitetura == 'SOM':
        return MapaSOM(config.som_linhas, config.som_colunas)
    if config.arquitetura == 'MLP':
        return nn.Sequential(nn.Flatten(), nn.Linear(784, config.neuronios),
                             nn.ReLU(), nn.Linear(config.neuronios, 10))
    if config.arquitetura == 'LeNet':
        # Variante LeNet-5: pooling médio fixo, convoluções completas e logits.
        # Borda preta de dois pixels: 28x28 -> 32x32, sem redimensionar o dígito.
        return nn.Sequential(
            nn.ZeroPad2d(2),
            nn.Conv2d(1, 6, 5), nn.Tanh(), nn.AvgPool2d(2),
            nn.Conv2d(6, 16, 5), nn.Tanh(), nn.AvgPool2d(2),
            nn.Conv2d(16, 120, 5), nn.Tanh(), nn.Flatten(),
            nn.Linear(120, 84), nn.Tanh(), nn.Linear(84, 10))
    if config.arquitetura != 'CNN':
        raise ValueError('Arquitetura desconhecida.')
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(32 * 7 * 7, config.neuronios),
        nn.ReLU(), nn.Linear(config.neuronios, 10))


def treinar(config, informar=lambda **kw: None, dados=None, destino=None):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    config.validar()
    inicio = time.monotonic()
    informar(fase='Carregando o MNIST. O primeiro download pode levar alguns minutos.')
    X, y = carregar() if dados is None else dados
    treino, validacao, teste = separar_tres(y, config)
    torch.set_num_threads(2)
    torch.manual_seed(config.semente)
    torch.use_deterministic_algorithms(True)
    dataset = TensorDataset(torch.from_numpy(X[treino].reshape(-1, 1, 28, 28)),
                            torch.from_numpy(y[treino]))
    loader = DataLoader(dataset, batch_size=config.lote, shuffle=True,
                        generator=torch.Generator().manual_seed(config.semente))
    pasta = Path(destino or BASE / 'resultados') / datetime.now().strftime('r%Y%m%d_%H%M%S_%f')
    pasta.mkdir(parents=True)
    (pasta / 'config.json').write_text(json.dumps(asdict(config), indent=2), encoding='utf-8')
    codigo = pasta / 'codigo_fonte'
    codigo.mkdir()
    for nome in ('modelo.py', 'modelos_classicos.py', 'arquiteturas.py', 'app.py',
                 'interface.html', 'requirements.txt', 'requirements.lock.txt',
                 'README.md', 'iniciar_linux.sh', 'comparar_cnn.py', 'executar_treino.py', 'instalar.py'):
        fonte = BASE / nome
        if fonte.exists():
            shutil.copy2(fonte, codigo / nome)
    np.savez_compressed(pasta / 'divisao.npz', treino=treino, validacao=validacao, teste=teste)
    informar(pasta_execucao=str(pasta))
    rede = construir_rede(config)
    perdas = []
    acuracias_treino = []
    perdas_validacao, acuracias_validacao = [], []
    melhor_perda, melhor_epoca, melhores_pesos = float('inf'), None, None
    # Limita concorrência das bibliotecas numéricas para manter o painel responsivo.
    with threadpool_limits(limits=2):
        if config.arquitetura == 'SOM':
            from modelos_classicos import ajustar_som
            informar(fase='Organizando o mapa de Kohonen com as imagens de treino')
            def progresso_som(epoca, historico):
                informar(fase=f'Organizando mapa: época {epoca} de {config.epocas}',
                         epoca=epoca, perdas=historico.copy(), segundos=time.monotonic() - inicio)
            perdas = ajustar_som(rede, dataset.tensors[0].flatten(1), dataset.tensors[1], config, progresso_som)
        else:
            if config.arquitetura == 'RBF':
                from sklearn.cluster import MiniBatchKMeans
                informar(fase='RBF: definindo centros gaussianos somente nas imagens de treino')
                centros = MiniBatchKMeans(n_clusters=config.neuronios, random_state=config.semente,
                                          batch_size=max(1024, config.neuronios), n_init=3,
                                          max_iter=20).fit(X[treino]).cluster_centers_
                rede.centros.copy_(torch.from_numpy(centros))
            otimizador = torch.optim.Adam(rede.parameters(), lr=config.taxa)
            criterio = torch.nn.CrossEntropyLoss()
            for epoca in range(1, config.epocas + 1):
                informar(fase=f'Treinando época {epoca} de {config.epocas}')
                rede.train()
                soma = 0.0
                acertos_treino = 0
                for imagens, rotulos in loader:
                    otimizador.zero_grad()
                    logits = rede(imagens)
                    erro = criterio(logits, rotulos)
                    erro.backward()
                    otimizador.step()
                    soma += erro.item() * len(rotulos)
                    acertos_treino += int((logits.detach().argmax(1) == rotulos).sum())
                perda = soma / len(treino)
                if not np.isfinite(perda):
                    raise ValueError('O treino divergiu. Reduza a taxa de aprendizado.')
                perdas.append(perda)
                acuracias_treino.append(acertos_treino / len(treino))
                if len(validacao):
                    rede.eval()
                    soma_val, acertos_val = 0.0, 0
                    with torch.inference_mode():
                        for inicio_val in range(0, len(validacao), config.lote):
                            indices = validacao[inicio_val:inicio_val + config.lote]
                            logits = rede(torch.from_numpy(X[indices].reshape(-1, 1, 28, 28)))
                            alvos = torch.from_numpy(y[indices])
                            soma_val += criterio(logits, alvos).item() * len(indices)
                            acertos_val += int((logits.argmax(1) == alvos).sum())
                    perda_val = soma_val / len(validacao)
                    if not np.isfinite(perda_val):
                        raise ValueError('A perda de validação não é finita.')
                    perdas_validacao.append(perda_val)
                    acuracias_validacao.append(acertos_val / len(validacao))
                    if perda_val < melhor_perda:
                        melhor_perda, melhor_epoca = perda_val, epoca
                        melhores_pesos = {k: v.detach().clone() for k, v in rede.state_dict().items()}
                informar(epoca=epoca, perdas=perdas.copy(),
                         perdas_validacao=perdas_validacao.copy(),
                         acuracias_treino=acuracias_treino.copy(),
                         acuracias_validacao=acuracias_validacao.copy(),
                         segundos=time.monotonic() - inicio)
        if melhores_pesos is not None:
            rede.load_state_dict(melhores_pesos)
        informar(fase='Avaliando nas imagens reservadas para teste')
        rede.eval()
        previsoes = []
        with torch.inference_mode():
            for inicio_lote in range(0, len(teste), config.lote):
                indices = teste[inicio_lote:inicio_lote + config.lote]
                imagens = torch.from_numpy(X[indices].reshape(-1, 1, 28, 28))
                previsoes.append(rede(imagens).argmax(dim=1).numpy())
        previstos = np.concatenate(previsoes)
    resultado = {
        'acuracia': float(accuracy_score(y[teste], previstos)),
        'acertos': int(np.sum(y[teste] == previstos)),
        'matriz': confusion_matrix(y[teste], previstos, labels=np.arange(10)).tolist(),
        'config': asdict(config), 'perdas': perdas,
        'perdas_validacao': perdas_validacao,
        'acuracias_validacao': acuracias_validacao,
        'acuracias_treino': acuracias_treino,
        'melhor_epoca': melhor_epoca,
        'relatorio_classes': classification_report(y[teste], previstos,
            labels=np.arange(10), output_dict=True, zero_division=0),
        'parametros': sum(p.numel() for p in rede.parameters()),
        'metrica_treino': 'Erro de quantização' if config.arquitetura == 'SOM' else 'Entropia cruzada',
        'segundos': time.monotonic() - inicio, 'pasta': str(pasta),
        'nao_utilizadas': len(y) - len(treino) - len(validacao) - len(teste),
        'exemplos': [{'real': int(y[i]), 'previsto': int(p),
                      'pixels': np.rint(X[i] * 255).astype(int).tolist()}
                     for i, p in zip(teste[:12], previstos[:12])],
    }
    if config.arquitetura == 'SOM':
        resultado['mapa_som'] = {'linhas': config.som_linhas, 'colunas': config.som_colunas,
                                 'rotulos': rede.rotulos.tolist(),
                                 'ocupacao': rede.contagens.sum(1).tolist()}
    if config.arquitetura in NOVAS_CNN:
        from arquiteturas import descrever
        resultado['arquitetura_detalhes'] = descrever(config.arquitetura, config.perfil_cnn)
    torch.save({'pesos': rede.state_dict(), 'config': asdict(config),
                'normalizacao': 'pixels / 255.0', 'melhor_epoca': melhor_epoca,
                'arquitetura_detalhes': resultado.get('arquitetura_detalhes')}, pasta / 'modelo.pt')
    np.savez_compressed(pasta / 'previsoes_teste.npz', indices=teste,
                        reais=y[teste], previstos=previstos)
    (pasta / 'resultado.json').write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding='utf-8')
    return resultado
