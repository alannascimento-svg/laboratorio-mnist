"""Famílias CNN para MNIST, treinadas do zero, sem pesos externos.

O perfil compacto reduz larguras/cabeças; o clássico conserva as larguras
usuais. Ambos adaptam a entrada para um canal e a saída para dez dígitos.
As diferenças em relação às versões ImageNet são registradas no resultado.
"""
import torch
from torch import nn
from torch.nn import functional as F

NOVAS = ('AlexNet', 'VGG19', 'GoogLeNet', 'ResNet18', 'MobileNetV2')
PERFIS = ('compacto', 'classico')


class BatchNormSegura(nn.BatchNorm2d):
    """Último lote unitário em mapas 1x1 usa estatísticas já acumuladas."""
    def forward(self, x):
        if self.training and x.numel() // x.shape[1] == 1:
            return F.batch_norm(x, self.running_mean, self.running_var,
                                self.weight, self.bias, False, 0.0, self.eps)
        return super().forward(x)


def conv_bn(a, b, kernel=3, stride=1, groups=1, relu6=False):
    return nn.Sequential(
        nn.Conv2d(a, b, kernel, stride, kernel // 2, groups=groups, bias=False),
        BatchNormSegura(b), nn.ReLU6(inplace=True) if relu6 else nn.ReLU(inplace=True))


class AlexNet(nn.Module):
    def __init__(self, compacto):
        super().__init__()
        canais = (32, 64, 128, 128, 64) if compacto else (64, 192, 384, 256, 256)
        a, b, c, d, e = canais
        self.features = nn.Sequential(
            nn.Conv2d(1, a, 3 if compacto else 11, stride=1 if compacto else 4,
                      padding=1 if compacto else 2), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2),
            nn.Conv2d(a, b, 5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2),
            nn.Conv2d(b, c, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c, d, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(d, e, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2))
        tamanho = 2 if compacto else 6
        h1, h2 = (256, 128) if compacto else (4096, 4096)
        self.pool = nn.AdaptiveAvgPool2d(tamanho)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(e * tamanho**2, h1), nn.ReLU(inplace=True),
            nn.Dropout(0.5), nn.Linear(h1, h2), nn.ReLU(inplace=True), nn.Linear(h2, 10))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)).flatten(1))


class VGG19(nn.Module):
    def __init__(self, compacto):
        super().__init__()
        canais = (16, 32, 64, 128, 128) if compacto else (64, 128, 256, 512, 512)
        camadas, entrada = [], 1
        # Configuração E: 2+2+4+4+4 convoluções, mais três camadas densas.
        for saida, quantidade in zip(canais, (2, 2, 4, 4, 4)):
            for _ in range(quantidade):
                camadas.extend([nn.Conv2d(entrada, saida, 3, padding=1), nn.ReLU(inplace=True)])
                entrada = saida
            camadas.append(nn.MaxPool2d(2, 2))
        self.features = nn.Sequential(*camadas)
        tamanho = 1 if compacto else 7
        h1, h2 = (256, 128) if compacto else (4096, 4096)
        self.pool = nn.AdaptiveAvgPool2d(tamanho)
        self.classifier = nn.Sequential(
            nn.Linear(canais[-1] * tamanho**2, h1), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(h1, h2), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(h2, 10))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)).flatten(1))


class Inception(nn.Module):
    def __init__(self, entrada, canais):
        super().__init__()
        a, b, c, d, e, f = canais
        self.branch1 = conv_bn(entrada, a, 1)
        self.branch3 = nn.Sequential(conv_bn(entrada, b, 1), conv_bn(b, c, 3))
        self.branch5 = nn.Sequential(conv_bn(entrada, d, 1), conv_bn(d, e, 5))
        self.branch_pool = nn.Sequential(nn.MaxPool2d(3, 1, padding=1), conv_bn(entrada, f, 1))
        self.saidas = a + c + e + f

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch3(x),
                          self.branch5(x), self.branch_pool(x)], dim=1)


class GoogLeNet(nn.Module):
    def __init__(self, compacto):
        super().__init__()
        escala = 0.25 if compacto else 1
        c = lambda n: max(4, int(n * escala))
        self.stem = nn.Sequential(
            conv_bn(1, c(64), 3 if compacto else 7, 1 if compacto else 2),
            nn.MaxPool2d(3, 2, padding=1), conv_bn(c(64), c(64), 1),
            conv_bn(c(64), c(192), 3), nn.MaxPool2d(3, 2, padding=1))
        specs = [(64, 96, 128, 16, 32, 32), (128, 128, 192, 32, 96, 64),
                 (192, 96, 208, 16, 48, 64), (160, 112, 224, 24, 64, 64),
                 (128, 128, 256, 24, 64, 64), (112, 144, 288, 32, 64, 64),
                 (256, 160, 320, 32, 128, 128), (256, 160, 320, 32, 128, 128),
                 (384, 192, 384, 48, 128, 128)]
        entrada, camadas = c(192), []
        for i, spec in enumerate(specs):
            bloco = Inception(entrada, tuple(c(n) for n in spec))
            camadas.append(bloco)
            entrada = bloco.saidas
            if i in (1, 6):
                camadas.append(nn.MaxPool2d(3, 2, padding=1))
        self.inceptions = nn.Sequential(*camadas)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(entrada, 10)

    def forward(self, x):
        x = self.pool(self.inceptions(self.stem(x))).flatten(1)
        return self.fc(self.dropout(x))


class BlocoResidual(nn.Module):
    def __init__(self, entrada, saida, stride=1):
        super().__init__()
        self.conv1 = conv_bn(entrada, saida, 3, stride)
        self.conv2 = nn.Sequential(nn.Conv2d(saida, saida, 3, padding=1, bias=False),
                                   BatchNormSegura(saida))
        self.skip = (nn.Identity() if entrada == saida and stride == 1 else
                     nn.Sequential(nn.Conv2d(entrada, saida, 1, stride, bias=False),
                                   BatchNormSegura(saida)))

    def forward(self, x):
        return F.relu(self.conv2(self.conv1(x)) + self.skip(x))


class ResNet18(nn.Module):
    def __init__(self, compacto):
        super().__init__()
        base = 16 if compacto else 64
        self.stem = nn.Sequential(conv_bn(1, base, 3 if compacto else 7,
                                          1 if compacto else 2),
                                  nn.Identity() if compacto else nn.MaxPool2d(3, 2, padding=1))
        entrada, camadas = base, []
        for etapa, saida in enumerate((base, base*2, base*4, base*8)):
            camadas.append(BlocoResidual(entrada, saida, 1 if etapa == 0 else 2))
            camadas.append(BlocoResidual(saida, saida))
            entrada = saida
        self.blocks = nn.Sequential(*camadas)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(entrada, 10)

    def forward(self, x):
        return self.fc(self.pool(self.blocks(self.stem(x))).flatten(1))


class ResidualInvertido(nn.Module):
    def __init__(self, entrada, saida, stride, expansao):
        super().__init__()
        oculto = entrada * expansao
        camadas = [] if expansao == 1 else [conv_bn(entrada, oculto, 1, relu6=True)]
        camadas.extend([conv_bn(oculto, oculto, 3, stride, groups=oculto, relu6=True),
                        nn.Conv2d(oculto, saida, 1, bias=False), BatchNormSegura(saida)])
        self.block = nn.Sequential(*camadas)
        self.usar_residual = stride == 1 and entrada == saida

    def forward(self, x):
        saida = self.block(x)
        return x + saida if self.usar_residual else saida


class MobileNetV2(nn.Module):
    def __init__(self, compacto):
        super().__init__()
        fator = 0.5 if compacto else 1
        c = lambda n: max(8, int(n * fator + 4) // 8 * 8)
        entrada = c(32)
        camadas = [conv_bn(1, entrada, 3, 2, relu6=True)]
        for expansao, canais, repeticoes, stride in [(1,16,1,1), (6,24,2,2),
                (6,32,3,2), (6,64,4,2), (6,96,3,1), (6,160,3,2), (6,320,1,1)]:
            saida = c(canais)
            for i in range(repeticoes):
                camadas.append(ResidualInvertido(entrada, saida, stride if i == 0 else 1, expansao))
                entrada = saida
        final = 640 if compacto else 1280
        camadas.append(conv_bn(entrada, final, 1, relu6=True))
        self.features = nn.Sequential(*camadas)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(final, 10))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)).flatten(1))


class RedeImagem(nn.Module):
    """Redimensionamento interno evita materializar 70 mil imagens maiores."""
    def __init__(self, arquitetura, perfil):
        super().__init__()
        compacto = perfil == 'compacto'
        self.tamanho = 32 if compacto or arquitetura == 'VGG19' else 64
        self.rede = globals()[arquitetura](compacto)
        for modulo in self.rede.modules():
            if isinstance(modulo, nn.Conv2d):
                nn.init.kaiming_normal_(modulo.weight, mode='fan_out', nonlinearity='relu')
                if modulo.bias is not None:
                    nn.init.zeros_(modulo.bias)
            elif isinstance(modulo, nn.Linear):
                # Cabeças pequenas precisam preservar o sinal de gradiente;
                # a inicialização 0,01 da VGG ImageNet usa cabeças muito maiores.
                if compacto:
                    nn.init.kaiming_normal_(modulo.weight, nonlinearity='relu')
                else:
                    nn.init.normal_(modulo.weight, 0, 0.01)
                nn.init.zeros_(modulo.bias)

    def forward(self, x):
        x = F.interpolate(x, size=(self.tamanho, self.tamanho), mode='bilinear',
                          align_corners=False, antialias=True)
        return self.rede(x)


def descrever(arquitetura, perfil):
    compacto = perfil == 'compacto'
    tamanho = 32 if compacto or arquitetura == 'VGG19' else 64
    estruturas = {
        'AlexNet': '5 convoluções, ReLU, max-pooling, dropout e 3 camadas densas',
        'VGG19': '16 convoluções 3×3 em blocos 2/2/4/4/4, 5 max-poolings e 3 camadas densas',
        'GoogLeNet': '9 módulos Inception com ramos 1×1, 3×3, 5×5 e pooling; normalização por lote, sem cabeças auxiliares',
        'ResNet18': '8 blocos residuais, duas convoluções por bloco, conexão de atalho e normalização por lote',
        'MobileNetV2': '17 blocos de resíduos invertidos, convoluções depthwise, ReLU6 e projeções lineares',
    }
    return {'familia': arquitetura, 'perfil': perfil, 'entrada': [1, tamanho, tamanho],
            'saida': 10, 'estrutura': estruturas[arquitetura], 'pesos_pre_treinados': False,
            'adaptacao': ('Larguras e cabeças reduzidas para MNIST.' if compacto else
                          'Larguras clássicas; resolução de entrada reduzida para MNIST.') +
                         ' Um canal, dez classes e redimensionamento bilinear interno.'}


def construir(arquitetura, perfil):
    if arquitetura not in NOVAS or perfil not in PERFIS:
        raise ValueError('Arquitetura ou perfil CNN desconhecido.')
    return RedeImagem(arquitetura, perfil)
