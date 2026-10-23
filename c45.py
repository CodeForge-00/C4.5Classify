import numpy as np
from Tree import Tree
from cutpoint import cutpoint
from pruning import makePruning


class C45:
    """
    Arbol de decision C4.5 para atributos continuos. La ultima columna del DataFrame es la clase.

    - En cada nodo se busca, para cada atributo, el mejor umbral binario (<= t / > t), de modo que
      un mismo atributo puede volver a dividirse mas abajo con otro umbral.
    - Entre los atributos con ganancia >= ganancia promedio se elige el de mayor gain ratio.
    - Post-poda por reemplazo de subarboles, con la variante elegida en `pruning` (ver pruning.py):
        'pessimistic': poda pesimista de C4.5 (parametro `cf`, por defecto 0.25).
        'penalty':     error de entrenamiento + omega * hojas / N (parametro `omega`).
      Tambien se puede pasar una instancia de PessimisticPruning / PenaltyPruning.
    """

    def __init__(self, min_objs=2, pruning='pessimistic', cf=0.25, omega=1.0):
        self.min_objs = min_objs
        self.pruning = makePruning(pruning, cf=cf, omega=omega)
        self.cut = cutpoint(min_objs)
        self.tree = Tree()

    # ---------------------------------------------------------------- construccion
    def fit(self, data, classes=None):
        self.columns = list(data.columns)
        self.features = self.columns[:-1]
        y_raw = data[self.columns[-1]].values
        if classes is None:
            classes = sorted(set(y_raw))
        self.classes = [c.item() if hasattr(c, 'item') else c for c in classes]
        lookup = {c: i for i, c in enumerate(self.classes)}
        y = np.array([lookup[v.item() if hasattr(v, 'item') else v] for v in y_raw])
        X = data[self.features].values.astype(float)
        self.total_instances = len(data)
        self.tree = Tree()
        self._build(X, y, None)
        return self

    def _leafData(self, counts):
        best = int(np.argmax(counts))
        return {
            'tag_name': self.classes[best],
            'classify_error': 1 - counts[best] / counts.sum(),
            'resumen_clases': [{'name': self.classes[i], 'count': int(c)}
                               for i, c in enumerate(counts) if c > 0],
        }

    def _build(self, X, y, parent):
        counts = np.bincount(y, minlength=len(self.classes))
        data = self._leafData(counts)
        split = None
        if np.count_nonzero(counts) > 1 and len(y) >= 2 * self.min_objs:
            split = self._chooseSplit(X, y)
        if split is None:
            self.tree.add(data, parent)
            return
        index, threshold = split
        data.update({
            'name': self.features[index],
            'index': index,
            'threshold': threshold,
            'labels': ['<=' + str(threshold), '>' + str(threshold)],
        })
        node = self.tree.add(data, parent)
        mask = X[:, index] <= threshold
        self._build(X[mask], y[mask], node)
        self._build(X[~mask], y[~mask], node)

    def _chooseSplit(self, X, y):
        candidates = []
        for j in range(X.shape[1]):
            result = self.cut.bestCut(X[:, j], y, len(self.classes))
            if result is not None:
                result['index'] = j
                candidates.append(result)
        if not candidates:
            return None
        average_gain = sum(c['gain'] for c in candidates) / len(candidates)
        eligible = [c for c in candidates if c['gain'] >= average_gain - 1e-12]
        best = max(eligible, key=lambda c: c['gain_ratio'])
        return best['index'], best['threshold']

    # ---------------------------------------------------------------- errores
    @staticmethod
    def _leafErrors(node):
        counts = [obj['count'] for obj in node.data['resumen_clases']]
        return sum(counts) - max(counts)

    def errorTotal(self):
        """Instancias de entrenamiento mal clasificadas por las hojas del arbol."""
        if self.tree.root is None:
            return 0
        return self._errorTotalAux(self.tree.root)

    def _errorTotalAux(self, nodo):
        if nodo.esHoja():
            return self._leafErrors(nodo)
        return sum(self._errorTotalAux(hijo) for hijo in nodo.childs)

    def trainingError(self):
        return self.errorTotal() / self.total_instances

    def generalizationError(self):
        return self.pruning.generalizationError(self)

    def treeError(self, pruning):
        """Suma de pruning.nodeError sobre las hojas del arbol."""
        if self.tree.root is None:
            return 0
        return self._treeErrorAux(self.tree.root, pruning)

    def _treeErrorAux(self, nodo, pruning):
        if nodo.esHoja():
            return self._nodeError(nodo, pruning)
        return sum(self._treeErrorAux(hijo, pruning) for hijo in nodo.childs)

    def getCantidadHojas(self):
        return self.tree.cantidadHojas()

    def getRules(self):
        return self.tree.getRules()

    # ---------------------------------------------------------------- post-poda
    def postPoda(self):
        if self.tree.root is not None:
            self._postPodaAux(self.tree.root)

    def _postPodaAux(self, nodo):
        """Devuelve (error, hojas) del subarbol ya podado segun el criterio de poda elegido."""
        if nodo.esHoja():
            return self._nodeError(nodo, self.pruning), 1
        errores = 0
        hojas = 0
        for hijo in nodo.childs:
            e, h = self._postPodaAux(hijo)
            errores += e
            hojas += h
        error_hoja = self._nodeError(nodo, self.pruning)
        if self.pruning.shouldPrune(error_hoja, errores, hojas):
            nodo.childs.clear()
            nodo.data = self._leafData(self._countsOf(nodo.data))
            return error_hoja, 1
        return errores, hojas

    @staticmethod
    def _nodeError(nodo, pruning):
        """Error del nodo si fuera una hoja (usa la distribucion de clases de sus instancias)."""
        counts = [obj['count'] for obj in nodo.data['resumen_clases']]
        return pruning.nodeError(sum(counts), sum(counts) - max(counts))

    def _countsOf(self, data):
        counts = np.zeros(len(self.classes), dtype=int)
        for obj in data['resumen_clases']:
            counts[self.classes.index(obj['name'])] = obj['count']
        return counts

    # ---------------------------------------------------------------- prediccion
    def predictClass(self, row, nodo=None):
        nodo = self.tree.root if nodo is None else nodo
        while not nodo.esHoja():
            if row[nodo.data['index']] <= nodo.data['threshold']:
                nodo = nodo.childs[0]
            else:
                nodo = nodo.childs[1]
        return nodo.data['tag_name']

    def predict(self, data):
        X = data[self.features].values.astype(float)
        return [self.predictClass(row) for row in X]

    def validationError(self, data):
        """Devuelve (error de validacion, matriz de confusion filas=real, columnas=predicha)."""
        expected = [v.item() if hasattr(v, 'item') else v for v in data[self.columns[-1]].values]
        predicted = self.predict(data)
        matriz = np.zeros((len(self.classes), len(self.classes)))
        errors = 0
        for real, pred in zip(expected, predicted):
            matriz[self.classes.index(real)][self.classes.index(pred)] += 1
            if real != pred:
                errors += 1
        return errors / len(expected), matriz
