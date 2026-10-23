import argparse
import numpy as np
from c45 import C45
from pruning import PessimisticPruning, PenaltyPruning
from training_set import Subsets

METRICS = ['training_error', 'generalization_error', 'validation_error', 'altura',
           'hojas_prepoda', 'hojas_postpoda']


def printMatrix(matriz):
    for fila in matriz:
        print(' '.join('{:.2f}'.format(valor) for valor in fila))


def run(pruning, k=10, seed=None, show_rules=True, verbose=True):
    """Validacion cruzada estratificada con la variante de poda `pruning` (nombre o instancia)."""
    sub = Subsets(seed=seed)
    classes = [int(c) for c in sorted(sub.load().iloc[:, -1].unique())]
    folds = sub.kFoldCrossValidation(k=k)

    results = []
    total_confusion = np.zeros((len(classes), len(classes)))
    for i, fold in enumerate(folds):
        algorithm = C45(pruning=pruning).fit(fold['training'], classes)
        hojas_prepoda = algorithm.getCantidadHojas()
        algorithm.postPoda()
        if verbose:
            print("Iteracion", i)
            if show_rules and i == 0:
                print("REGLAS")
                algorithm.getRules()
        validation_error, confusion = algorithm.validationError(fold['testing'])
        total_confusion += confusion
        results.append({'hojas_prepoda': hojas_prepoda,
                        'hojas_postpoda': algorithm.getCantidadHojas(),
                        'training_error': algorithm.trainingError(),
                        'generalization_error': algorithm.generalizationError(),
                        'validation_error': validation_error,
                        'altura': algorithm.tree.altura()})
        if verbose:
            print(results[-1])
            print("MATRIZ DE CONFUSION")
            printMatrix(confusion)

    average = {key: sum(r[key] for r in results) / len(results) for key in METRICS}
    if verbose:
        print("Resultado General")
        for key in METRICS:
            print(key, "promedio", average[key])
        print("MATRIZ DE CONFUSION (suma de todos los folds), clases", classes)
        printMatrix(total_confusion)
    return average, total_confusion


def main():
    parser = argparse.ArgumentParser(description='C4.5 sobre Wine Quality (red) con validacion cruzada')
    parser.add_argument('--pruning', choices=['pessimistic', 'penalty', 'both'], default='pessimistic')
    parser.add_argument('--cf', type=float, default=0.25, help='nivel de confianza de la poda pesimista')
    parser.add_argument('--omega', type=float, default=1.0, help='costo por hoja de la poda por penalidad')
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    variants = {'pessimistic': PessimisticPruning(args.cf), 'penalty': PenaltyPruning(args.omega)}
    if args.pruning != 'both':
        run(variants[args.pruning], k=args.k, seed=args.seed)
        return

    averages = {name: run(variant, k=args.k, seed=args.seed, verbose=False)[0]
                for name, variant in variants.items()}
    print('{:<22}{:>14}{:>14}'.format('promedio', 'pessimistic', 'penalty'))
    for key in METRICS:
        print('{:<22}{:>14.4f}{:>14.4f}'.format(key, averages['pessimistic'][key], averages['penalty'][key]))


if __name__ == '__main__':
    main()
