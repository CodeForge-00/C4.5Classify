import math
import numpy as np
from util import util


class cutpoint:
    """
    Busca el mejor punto de corte binario (<= t / > t) de un atributo continuo, tal como hace C4.5
    en cada nodo: se ordenan los valores, se prueban los umbrales entre valores distintos
    consecutivos y se elige el de mayor ganancia de informacion.

    Variantes de discretizacion que se probaron antes (ya no se usan, la discretizacion ahora se
    hace dentro de cada nodo):
      1. Usar la media: sensible a valores atipicos.
      2. Rangos con la misma cantidad de elementos (pd.qcut).
      3. Rangos con la misma longitud (pd.cut).
      4. Agrupar valores sucesivos de una misma clase: genera demasiadas etiquetas en datos continuos.
    """

    def __init__(self, min_objs=2):
        self.min_objs = min_objs

    def bestCut(self, values, classes, n_classes):
        """
        values: valores del atributo (1D). classes: indice de clase de cada instancia (enteros
        0..n_classes-1). Devuelve None si no hay un corte valido con ganancia positiva, o un
        diccionario con threshold, gain, split_info y gain_ratio.
        """
        n = len(values)
        order = np.argsort(values, kind='mergesort')
        v = values[order]
        c = classes[order]

        onehot = np.zeros((n, n_classes))
        onehot[np.arange(n), c] = 1
        cum = np.cumsum(onehot, axis=0)
        total = cum[-1]

        distinct = v[:-1] < v[1:]
        left_n = np.arange(1, n)
        valid = distinct & (left_n >= self.min_objs) & (n - left_n >= self.min_objs)
        if not valid.any():
            return None

        idx = np.nonzero(valid)[0]
        left = cum[idx]
        right = total - left
        ln = left_n[idx]
        rn = n - ln

        base = util.fatherEntropy(total)
        gain = base - (ln * util.entropy(left) + rn * util.entropy(right)) / n
        # Correccion de C4.5 release 8: penaliza atributos con muchos umbrales candidatos
        gain = gain - math.log2(max(int(distinct.sum()), 1)) / n

        best = int(np.argmax(gain))
        if gain[best] <= 1e-12:
            return None
        split_info = util.splitInfo([ln[best], rn[best]])
        return {
            'threshold': float(v[idx[best]]),
            'gain': float(gain[best]),
            'split_info': split_info,
            'gain_ratio': util.gainRatio(float(gain[best]), split_info),
        }
