"""RBF gaussiana e SOM de Kohonen em lote, com ajuste apenas no treino."""
import torch
from torch import nn


def distancias_quadradas(x, centros):
    return (x.square().sum(1, keepdim=True) + centros.square().sum(1)[None, :]
            - 2 * x @ centros.T).clamp_min(0)


class RedeRBF(nn.Module):
    def __init__(self, quantidade, sigma):
        super().__init__()
        self.register_buffer('centros', torch.zeros(quantidade, 784))
        self.register_buffer('sigma', torch.tensor(float(sigma)))
        self.saida = nn.Linear(quantidade, 10)

    def forward(self, imagens):
        distancias = distancias_quadradas(imagens.flatten(1), self.centros)
        ativacoes = torch.exp(-distancias / (2 * self.sigma.square()))
        return self.saida(ativacoes)


class MapaSOM(nn.Module):
    def __init__(self, linhas, colunas):
        super().__init__()
        self.linhas, self.colunas = linhas, colunas
        quantidade = linhas * colunas
        self.register_buffer('prototipos', torch.zeros(quantidade, 784))
        self.register_buffer('rotulos', torch.zeros(quantidade, dtype=torch.long))
        self.register_buffer('contagens', torch.zeros(quantidade, 10, dtype=torch.long))

    def vencedores(self, imagens):
        return distancias_quadradas(imagens.flatten(1), self.prototipos).argmin(1)

    def forward(self, imagens):
        # Indicadores de classe, não probabilidades calibradas.
        return torch.nn.functional.one_hot(self.rotulos[self.vencedores(imagens)], 10).float()


@torch.no_grad()
def ajustar_som(rede, X, y, config, informar_epoca):
    """SOM batch: w_j = sum_i h(j, BMU_i) x_i / sum_i h(j, BMU_i)."""
    n = len(X)
    unidades = len(rede.prototipos)
    gerador = torch.Generator().manual_seed(config.semente)
    indices = torch.randint(n, (unidades,), generator=gerador)
    rede.prototipos.copy_(X[indices])
    grade = torch.cartesian_prod(torch.arange(rede.linhas), torch.arange(rede.colunas)).float()
    dist_grade = distancias_quadradas(grade, grade)
    perdas = []
    for epoca in range(config.epocas):
        sigma = config.som_sigma * (0.5 / config.som_sigma) ** (epoca / max(config.epocas - 1, 1))
        vizinhanca = torch.exp(-dist_grade / (2 * sigma**2))
        somas = torch.zeros_like(rede.prototipos)
        contagens = torch.zeros(unidades)
        for inicio in range(0, n, config.lote):
            bloco = X[inicio:inicio + config.lote]
            bmu = rede.vencedores(bloco)
            somas.index_add_(0, bmu, bloco)
            contagens += torch.bincount(bmu, minlength=unidades)
        denominador = vizinhanca @ contagens
        novos = (vizinhanca @ somas) / denominador[:, None].clamp_min(1e-20)
        atualizaveis = denominador > 1e-20
        rede.prototipos[atualizaveis] = novos[atualizaveis]
        erro = 0.0
        for inicio in range(0, n, config.lote):
            bloco = X[inicio:inicio + config.lote]
            bmu = rede.vencedores(bloco)
            erro += (bloco - rede.prototipos[bmu]).norm(dim=1).sum().item()
        perdas.append(erro / n)
        informar_epoca(epoca + 1, perdas)

    # Associação supervisionada posterior: votos somente dos exemplos de treino.
    rede.contagens.zero_()
    for inicio in range(0, n, config.lote):
        bmu = rede.vencedores(X[inicio:inicio + config.lote])
        votos = torch.bincount(bmu * 10 + y[inicio:inicio + config.lote], minlength=unidades * 10)
        rede.contagens.add_(votos.reshape(unidades, 10))
    rede.rotulos.copy_(rede.contagens.argmax(1))
    ocupados = rede.contagens.sum(1) > 0
    # Unidades sem votos herdam a classe do protótipo ocupado mais próximo.
    if not ocupados.all():
        proximos = distancias_quadradas(rede.prototipos[~ocupados], rede.prototipos[ocupados]).argmin(1)
        rede.rotulos[~ocupados] = rede.rotulos[ocupados][proximos]
    return perdas
