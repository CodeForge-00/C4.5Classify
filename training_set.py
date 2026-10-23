import numpy as np
import pandas as pd

data_file = 'Data/winequality-red.csv'


class Subsets:
    """Genera particiones entrenamiento/prueba estratificadas por clase (en memoria)."""

    def __init__(self, file=data_file, seed=None):
        self.file = file
        self.rng = np.random.RandomState(seed)

    def load(self):
        return pd.read_csv(self.file)

    def RandomSubSampling(self, iterations=1, tag_index=-1, training_percent=0.8):
        """Hold-out repetido: cada iteracion mezcla de nuevo y respeta la proporcion de clases."""
        data = self.load()
        tags = data.iloc[:, tag_index].values
        result = []
        for _ in range(iterations):
            training = []
            for tag in np.unique(tags):
                members = self.rng.permutation(np.nonzero(tags == tag)[0])
                take = int(round(len(members) * training_percent))
                if len(members) > 1:
                    take = min(max(take, 1), len(members) - 1)
                training.extend(members[:take])
            mask = np.zeros(len(data), dtype=bool)
            mask[training] = True
            result.append({'training': data[mask].reset_index(drop=True),
                           'testing': data[~mask].reset_index(drop=True)})
        return result

    def kFoldCrossValidation(self, k=10, tag_index=-1):
        """
        k-fold estratificado: las instancias de cada clase se mezclan y se reparten por turnos
        entre los folds (el turno continua de una clase a la siguiente para que los folds queden
        del mismo tamano). Devuelve k pares {'training': df, 'testing': df}.
        """
        data = self.load()
        tags = data.iloc[:, tag_index].values
        fold_of = np.zeros(len(data), dtype=int)
        offset = 0
        for tag in np.unique(tags):
            members = self.rng.permutation(np.nonzero(tags == tag)[0])
            fold_of[members] = (offset + np.arange(len(members))) % k
            offset += len(members)

        folds = []
        for i in range(k):
            testing = fold_of == i
            folds.append({'training': data[~testing].reset_index(drop=True),
                          'testing': data[testing].reset_index(drop=True)})
        return folds
