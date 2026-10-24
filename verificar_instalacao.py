"""Verificação opcional com MNIST real; primeiro uso requer internet."""
from pathlib import Path
import tempfile

import numpy as np
import torch

from modelo import Config, carregar, construir_rede, treinar


def main():
    X, y = carregar()
    assert X.shape == (70000, 784)
    assert np.isfinite(X).all() and X.min() >= 0 and X.max() <= 1
    assert set(y) == set(range(10))
    with tempfile.TemporaryDirectory() as temporario:
        config = Config(arquitetura='LeNet', treino=200, validacao=100,
                        teste=100, epocas=1, lote=64)
        resultado = treinar(config, dados=(X, y), destino=temporario)
        pasta = Path(resultado['pasta'])
        pacote = torch.load(pasta / 'modelo.pt', weights_only=True)
        rede = construir_rede(Config(**pacote['config']))
        rede.load_state_dict(pacote['pesos'])
        rede.eval()
        with np.load(pasta / 'previsoes_teste.npz') as salvo:
            with torch.inference_mode():
                previstos = rede(torch.from_numpy(X[salvo['indices']].reshape(-1, 1, 28, 28))).argmax(1).numpy()
            np.testing.assert_array_equal(previstos, salvo['previstos'])
    print('MNIST baixado/carregado: 70.000 imagens válidas.')
    print('Treinamento curto e recarga: todas as 100 previsões reproduzidas.')
    print('A verificação mede funcionamento, não qualidade de classificação.')


if __name__ == '__main__':
    main()
