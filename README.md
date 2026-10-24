# Laboratório MNIST

Um laboratório local para experimentar redes neurais no reconhecimento de dígitos escritos à mão. O painel permite escolher a arquitetura, configurar treino, validação e teste, acompanhar as curvas e consultar os resultados salvos no próprio computador.

O projeto usa Python e PyTorch, funciona na CPU e não exige conta, credencial, chave de API ou GPU. O código é distribuído sob a licença [MIT](LICENSE). O conjunto MNIST é obtido de uma fonte pública e tem seus próprios termos de uso.

## Instalação

Use **Python 3.12**, recomendado. O instalador aceita Python **3.11 a 3.13** em Windows, Linux e macOS. É necessário acesso à internet para instalar os pacotes e baixar o MNIST pela primeira vez. Reserve espaço para o ambiente Python, a base e os resultados; os modelos clássicos maiores podem consumir bastante memória.

Baixe o ZIP deste repositório e extraia-o, ou use Git:

```text
git clone https://github.com/alannascimento-svg/laboratorio-mnist.git
cd laboratorio-mnist
```

Execute os comandos abaixo na pasta que contém este README. Não copie um ambiente `.venv` de outro computador.

### Windows

Instale o Python pelo [site oficial](https://www.python.org/downloads/), com o inicializador `py` ou a opção de adicionar Python ao PATH. No terminal, execute:

```bat
instalar_windows.bat
iniciar_windows.bat
```

### Linux e macOS

Instale Python 3.12 pelo gerenciador do sistema ou pelo [site oficial](https://www.python.org/downloads/). Em distribuições Linux que separam esse componente, instale também o suporte a `venv` da versão escolhida. Execute:

```bash
bash instalar_linux.sh
bash iniciar_linux.sh
```

Apesar do nome, os dois scripts também atendem ao macOS. O instalador procura Python 3.12, 3.13 ou 3.11, cria `.venv` e instala as dependências. Em Linux e Windows, instala primeiro PyTorch pelo índice oficial de pacotes para CPU; no macOS, utiliza o PyPI. A aplicação usa CPU em todos os sistemas.

### Instalação manual e verificação

Também é possível chamar o instalador diretamente com um Python compatível:

```text
python instalar.py
```

Em Linux/macOS, o comando pode ser `python3.12 instalar.py`. O arquivo `requirements.txt` lista todas as dependências diretas, inclusive PyTorch. O instalador escolhe a origem de PyTorch apropriada e verifica os pacotes depois da instalação; não há dependência de programas ou caminhos de outro computador.

Para conferir o ambiente sem instalar nada:

```text
# Windows
.venv\Scripts\python.exe instalar.py --verificar

# Linux/macOS
.venv/bin/python instalar.py --verificar
```

Se a instalação for interrompida, execute o instalador novamente. Se `.venv` tiver sido copiada ou criada com uma versão de Python incompatível, remova somente essa pasta e reinstale. Os dados e os resultados ficam em outras pastas.

## Primeiro experimento

Abra **[http://127.0.0.1:8503](http://127.0.0.1:8503)** depois de iniciar o painel. O terminal deve permanecer aberto. O endereço funciona somente no computador que está executando o programa; cada colega instala sua própria cópia.

Comece com a seguinte configuração, especialmente em computadores sem GPU:

| Campo | Valor inicial sugerido |
|---|---:|
| Rede | LeNet-5 |
| Imagens para treino | 2.000 |
| Imagens para validação | 500 |
| Imagens para teste | 500 |
| Épocas | 3 |
| Tamanho do lote | 64 |
| Taxa de aprendizado | 0,001 |
| Semente | 42 |
| Perfil das novas CNNs | Compacto |

Clique em **Iniciar treinamento**. No primeiro uso, o download da base completa pode levar alguns minutos, mesmo com poucas imagens selecionadas. Os próximos usos reutilizam o cache local. A página não começa a treinar sozinha.

Use **Parar treinamento** para solicitar a interrupção de uma execução demorada. O treinamento roda em um processo separado, que é encerrado pelo botão, mantendo o painel aberto. Não há retomada a partir de uma execução interrompida; os resultados de treinos anteriores continuam disponíveis. Para encerrar o servidor inteiro, use `Ctrl+C` no terminal.

Para trocar a porta:

```text
# Windows
iniciar_windows.bat --port 8504

# Linux/macOS
bash iniciar_linux.sh --port 8504
```

Nesse caso, abra `http://127.0.0.1:8504`. O sistema foi preparado para estudo local; publicá-lo no GitHub disponibiliza o código para instalação, não um servidor compartilhado de treinamento.

## Arquiteturas disponíveis

As dez opções são treinadas do zero. Nenhuma baixa pesos de ImageNet ou exige `torchvision`.

| Opção | Implementação neste laboratório |
|---|---|
| MLP | 784 entradas, camada oculta ReLU configurável e dez saídas. |
| CNN | Duas convoluções, max-pooling e camada densa configurável. |
| LeNet-5 | Borda de dois pixels, convoluções 5×5 com 6/16/120 filtros, Tanh, pooling médio e camada densa de 84 unidades. |
| AlexNet | Cinco convoluções e três camadas densas, com dropout. |
| VGG19 | Dezesseis convoluções em cinco blocos e três camadas densas. |
| GoogLeNet | Nove módulos Inception; não inclui cabeças auxiliares. |
| ResNet-18 | Oito blocos residuais com duas convoluções cada. |
| MobileNetV2 | Dezessete blocos de resíduos invertidos, convoluções por canal e ReLU6. |
| RBF | Centros gaussianos ajustados somente no treino; saída linear supervisionada. |
| SOM | Mapa de Kohonen em lote, organizado sem rótulos; os rótulos de treino definem as classes após a organização. |

**São adaptações didáticas**, e não reproduções exatas de todos os detalhes dos artigos originais. Todos os modelos recebem um canal e classificam dez dígitos.

Nas cinco famílias CNN adicionadas, o perfil **Compacto** reduz filtros e cabeças e usa imagens 32×32, redimensionadas internamente. O perfil **Clássico** usa larguras maiores, ainda com adaptações para MNIST e resoluções reduzidas: 64×64, exceto VGG19, que usa 32×32. As camadas densas de AlexNet e VGG19 no perfil clássico têm 4.096 unidades e podem tornar o treino muito lento na CPU. Comece pelo perfil compacto.

A LeNet usa conexões completas nas convoluções, pooling médio fixo e dez logits com entropia cruzada. Não reproduz as conexões parciais, o subsampling treinável e a saída por distâncias da rede original. A GoogLeNet usa normalização por lote e omite as cabeças auxiliares. A VGG19 implementada não usa normalização por lote. Os detalhes das adaptações estão em `arquiteturas.py` e são registrados nos resultados.

MLP, CNN, LeNet, novas CNNs e RBF usam Adam e entropia cruzada. Na RBF, os centros são definidos por MiniBatchKMeans no treino e permanecem fixos; as épocas ajustam a camada de saída. O SOM usa erro de quantização, que não deve ser comparado numericamente à perda dos modelos supervisionados. SOM não utiliza conjunto de validação: selecione zero imagens para validação nessa opção.

## Dados e comparação de experimentos

O programa baixa a base pública **[MNIST pelo OpenML](https://www.openml.org/d/554)** (`mnist_784`, versão 1), com 70.000 imagens 28×28 de dígitos de 0 a 9. A função `carregar()` armazena o cache em `dados/` e divide os pixels por 255, sem ajuste estatístico. A instalação e o treinamento não pedem credenciais.

O repositório não inclui imagens locais, bases particulares, ambientes Python, modelos já treinados nem históricos pessoais. O primeiro download da base requer internet; depois de concluído, o cache permite repetir experimentos sem baixar novamente.

A divisão deste projeto é **aleatória e estratificada**, com semente configurável e índices distintos para treino, validação e teste. A soma pode ser menor que 70.000; o restante não participa do experimento. **Essa divisão difere da divisão oficial de 60.000 imagens de treino e 10.000 de teste do MNIST.** Portanto, os resultados não são diretamente comparáveis ao benchmark oficial.

Nos modelos supervisionados, a menor perda de validação escolhe os pesos salvos. Todas as épocas solicitadas são executadas, salvo interrupção. Com validação zero, salva-se a última época. As imagens de teste são utilizadas somente para a avaliação final.

Para comparar arquiteturas, mantenha quantidades, semente e demais condições e confira os índices de `divisao.npz`. A semente e as opções determinísticas ajudam a repetir execuções, mas versões de bibliotecas e sistemas diferentes podem produzir diferenças numéricas. Reutilizar o mesmo teste para orientar sucessivos ajustes torna a comparação exploratória.

Treinos de três épocas e poucas imagens servem para verificar o funcionamento e aprender a usar o painel. Acurácias baixas em ensaios curtos, especialmente em arquiteturas profundas, não permitem concluir que uma família é inferior. Uma comparação acadêmica exige protocolo documentado, ajuste de hiperparâmetros, tempo de treinamento adequado e avaliação consistente.

## Resultados e arquivos locais

Cada treino concluído cria uma pasta em `resultados/`, com configuração, índices das partições, pesos, métricas por classe, matriz de confusão, curvas, previsões e cópia do código utilizado. O painel mostra exemplos de teste e mantém um histórico das execuções concluídas. O SOM também apresenta classes e ocupação do mapa.

| Pasta | Conteúdo |
|---|---|
| `.venv/` | Ambiente Python criado pelo instalador. |
| `dados/` | Cache do MNIST público baixado pelo programa. |
| `resultados/` | Modelos e arquivos das execuções locais. |
| `comparacoes/` | Tabelas de comparações executadas pelo terminal. |

Essas pastas são locais e estão excluídas do versionamento. Não adicione credenciais, arquivos `.env`, bases particulares ou resultados com informações que não devam ser públicos. Ao compartilhar um resultado, revise o conteúdo dos arquivos, inclusive caminhos locais presentes nos metadados.

## Comparação pelo terminal

Este comando compara LeNet e as cinco novas famílias com os mesmos índices, no perfil compacto:

```text
# Windows
.venv\Scripts\python.exe comparar_cnn.py --perfil compacto --treino 2000 --validacao 500 --teste 500 --epocas 3 --lote 64 --semente 42

# Linux/macOS
.venv/bin/python comparar_cnn.py --perfil compacto --treino 2000 --validacao 500 --teste 500 --epocas 3 --lote 64 --semente 42
```

Os resultados individuais ficam em `resultados/`, e o resumo CSV/JSON fica em `comparacoes/`. Para verificar uma única arquitetura, acrescente, por exemplo, `--arquiteturas LeNet`. Interrompa essa execução de terminal com `Ctrl+C`.

## Testes

Após instalar, execute a suíte na pasta do projeto:

```text
# Windows
.venv\Scripts\python.exe -m unittest discover -p "test_*.py" -v

# Linux/macOS
.venv/bin/python -m unittest discover -p 'test_*.py' -v
```

Os testes usam dados sintéticos e pastas temporárias para verificar as arquiteturas, partições, gradientes, validação e recarga dos modelos. Não precisam do download do MNIST. Não substituem um experimento com imagens reais, nem comprovam a acurácia de uma arquitetura. Alguns testes constroem modelos no perfil clássico e podem exigir mais memória que o primeiro experimento sugerido.

Para uma verificação adicional com imagens reais, execute `verificar_instalacao.py` com o Python da `.venv`. Esse comando baixa/carrega o MNIST, faz um treinamento mínimo em uma pasta temporária e verifica se a recarga reproduz as previsões. Não é um benchmark de acurácia.

A instalação limpa e os 19 testes foram verificados em Linux, com Python 3.12.14 e PyTorch 2.14.1 para CPU, em 24/10/2026. O workflow do GitHub também executa a suíte em Windows e macOS; o resultado de cada execução pode ser consultado na aba Actions.

## Referências

- [LeCun et al. — Gradient-Based Learning Applied to Document Recognition](https://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf).
- Implementações de referência do PyTorch: [AlexNet](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.alexnet.html), [VGG19](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vgg19.html), [GoogLeNet](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.googlenet.html), [ResNet-18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) e [MobileNetV2](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v2.html).
- [MiniBatchKMeans no scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html).
- [SOM em lote — SOM Toolbox](https://www.cis.hut.fi/somtoolbox/package/docs2/som_batchtrain.html).
