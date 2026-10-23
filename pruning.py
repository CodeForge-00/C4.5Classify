import math


class Pruning:
    """
    Criterio de post-poda por reemplazo de subarboles. C45 recorre el arbol de abajo hacia arriba y,
    en cada nodo interno, convierte el subarbol en una hoja si `shouldPrune` lo aprueba.

    Cada variante implementa:
      nodeError(n, e)            error asignado a un nodo con n instancias y e mal clasificadas
                                 (se usa para hojas y para el nodo si se volviera hoja).
      shouldPrune(...)           decide si el nodo pasa a ser hoja.
      generalizationError(model) estimacion del error de generalizacion del arbol completo.
    """
    name = None

    def nodeError(self, n, e):
        raise NotImplementedError

    def shouldPrune(self, leaf_error, subtree_error, subtree_leaves):
        raise NotImplementedError

    def generalizationError(self, model):
        raise NotImplementedError


class PessimisticPruning(Pruning):
    """
    Poda pesimista de C4.5: el error de cada hoja es el limite superior, con nivel de confianza
    `cf`, del error binomial (funcion AddErrs de Quinlan). Un nodo se poda si el error pesimista
    como hoja es <= la suma de los errores pesimistas de sus hojas + 0.1. Un `cf` menor poda mas.
    """
    name = 'pessimistic'

    def __init__(self, cf=0.25):
        self.cf = cf
        self.z2 = self._normalDeviate(cf) ** 2

    @staticmethod
    def _normalDeviate(cf):
        """z tal que P(Z > z) = cf para una normal estandar (biseccion, sin scipy)."""
        lo, hi = 0.0, 10.0
        for _ in range(100):
            mid = (lo + hi) / 2
            if 0.5 * math.erfc(mid / math.sqrt(2)) > cf:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    def addErrs(self, n, e):
        """Errores extra que C4.5 suma a los `e` observados en `n` instancias."""
        if n <= 0:
            return 0.0
        if e < 1e-6:
            return n * (1 - math.exp(math.log(self.cf) / n))
        if e < 0.9999:
            base = n * (1 - math.exp(math.log(self.cf) / n))
            return base + e * (self.addErrs(n, 1) - base)
        if e + 0.5 >= n:
            return 0.67 * (n - e)
        z2 = self.z2
        pr = (e + 0.5 + z2 / 2 + math.sqrt(z2 * ((e + 0.5) * (1 - (e + 0.5) / n) + z2 / 4))) / (n + z2)
        return n * pr - e

    def nodeError(self, n, e):
        return e + self.addErrs(n, e)

    def shouldPrune(self, leaf_error, subtree_error, subtree_leaves):
        return leaf_error <= subtree_error + 0.1

    def generalizationError(self, model):
        """Suma de errores pesimistas de las hojas dividida entre N."""
        if model.tree.root is None:
            return 0.0
        return model.treeError(self) / model.total_instances


class PenaltyPruning(Pruning):
    """
    Poda por penalidad: error de generalizacion  e_train + omega * hojas / N. Un nodo se poda si
    eso no empeora, es decir  errores_hoja + omega <= errores_subarbol + omega * hojas_subarbol.
    Un `omega` mayor poda mas.
    """
    name = 'penalty'

    def __init__(self, omega=1.0):
        self.omega = omega

    def nodeError(self, n, e):
        return e

    def shouldPrune(self, leaf_error, subtree_error, subtree_leaves):
        return leaf_error + self.omega <= subtree_error + self.omega * subtree_leaves

    def generalizationError(self, model):
        return model.trainingError() + self.omega * model.getCantidadHojas() / model.total_instances


def makePruning(pruning='pessimistic', cf=0.25, omega=1.0):
    """Acepta el nombre de la variante ('pessimistic' / 'penalty') o una instancia de Pruning."""
    if isinstance(pruning, Pruning):
        return pruning
    if pruning == 'pessimistic':
        return PessimisticPruning(cf)
    if pruning == 'penalty':
        return PenaltyPruning(omega)
    raise ValueError("pruning debe ser 'pessimistic', 'penalty' o una instancia de Pruning")
