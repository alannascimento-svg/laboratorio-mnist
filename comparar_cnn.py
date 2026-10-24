"""Compara CNNs na mesma divisão; padrões são um ensaio curto de funcionamento."""
import argparse
import csv
import json
from dataclasses import asdict
from datetime import datetime

import numpy as np

from modelo import BASE, Config, NOVAS_CNN, carregar, treinar


def comparar(args):
    dados = carregar()
    destino = BASE / 'comparacoes' / datetime.now().strftime('cnn_%Y%m%d_%H%M%S_%f')
    destino.mkdir(parents=True)
    linhas, configuracoes, referencia = [], [], None
    (destino / 'config_comparacao.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')
    for nome in args.arquiteturas:
        config = Config(arquitetura=nome, perfil_cnn=args.perfil, treino=args.treino,
                        validacao=args.validacao, teste=args.teste, epocas=args.epocas,
                        lote=args.lote, taxa=args.taxa, semente=args.semente)
        config.validar()
        configuracoes.append(asdict(config))
        print(f'Iniciando {nome}: {args.epocas} épocas, perfil {args.perfil}', flush=True)

        def progresso(**evento):
            if 'epoca' in evento:
                print(f"{nome}: época {evento['epoca']}/{args.epocas}; "
                      f"perda {evento['perdas'][-1]:.4f}; "
                      f"validação {evento['acuracias_validacao'][-1]:.2%}" if args.validacao else
                      f"{nome}: época {evento['epoca']}/{args.epocas}", flush=True)

        resultado = treinar(config, dados=dados, informar=progresso)
        indices = np.load(resultado['pasta'] + '/divisao.npz')
        if referencia is None:
            referencia = {k: indices[k].copy() for k in indices.files}
        for k in referencia:
            np.testing.assert_array_equal(referencia[k], indices[k])
        indices.close()
        linhas.append({'arquitetura': nome, 'perfil': args.perfil if nome in NOVAS_CNN else 'fixo',
                       'treino': args.treino, 'validacao': args.validacao, 'teste': args.teste,
                       'epocas': args.epocas, 'melhor_epoca': resultado['melhor_epoca'],
                       'acuracia': resultado['acuracia'],
                       'f1_macro': resultado['relatorio_classes']['macro avg']['f1-score'],
                       'parametros': resultado['parametros'], 'segundos': resultado['segundos'],
                       'pasta': resultado['pasta']})
        with (destino / 'resumo.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(linhas[0]))
            writer.writeheader()
            writer.writerows(linhas)
        (destino / 'comparacao.json').write_text(json.dumps(
            {'concluida': len(linhas) == len(args.arquiteturas),
             'particoes_identicas': True, 'configuracoes': configuracoes,
             'resultados': linhas,
             'observacao': 'Ensaio inicial; não constitui busca de hiperparâmetros ou ranking conclusivo.'},
            indent=2, ensure_ascii=False), encoding='utf-8')
        print(f"{nome}: acurácia {resultado['acuracia']:.2%}; {resultado['segundos']:.1f} s", flush=True)
    print(f'Comparação salva em: {destino}', flush=True)
    return destino


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arquiteturas', nargs='+', choices=('LeNet', *NOVAS_CNN),
                        default=['LeNet', *NOVAS_CNN])
    parser.add_argument('--perfil', choices=('compacto', 'classico'), default='compacto')
    for nome, padrao in [('treino', 2000), ('validacao', 500), ('teste', 500),
                         ('epocas', 3), ('lote', 64), ('semente', 42)]:
        parser.add_argument('--' + nome, type=int, default=padrao)
    parser.add_argument('--taxa', type=float, default=0.001)
    comparar(parser.parse_args())
