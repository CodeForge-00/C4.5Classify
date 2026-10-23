class TreeNode:
    def __init__(self, value):
        self.childs = []
        self.data = value

    def add(self, val):
        child = TreeNode(val)
        self.childs.append(child)
        return child

    def esHoja(self):
        return len(self.childs) == 0

    def info(self):
        if self.esHoja():
            return str(self.data['tag_name']) + ' (Classify Error: ' + str(self.data['classify_error']) + ')'
        return self.data['name']

    def getCantidadHijo(self):
        """Cantidad de descendientes (hijos, nietos, ...) del nodo."""
        suma = 0
        for obj in self.childs:
            suma += 1 + obj.getCantidadHijo()
        return suma


class Tree:
    def __init__(self):
        self.root = None

    def getRoot(self):
        return self.root

    def add(self, val, parent=None):
        """Agrega un nodo bajo `parent` (o como raiz si no hay) y lo devuelve."""
        if parent is None:
            if self.root is not None:
                raise ValueError('El arbol ya tiene raiz, indique el padre')
            self.root = TreeNode(val)
            return self.root
        return parent.add(val)

    def deleteTree(self):
        self.root = None

    def printTree(self):
        if self.root is not None:
            self._printTree(self.root)

    def _printTree(self, node):
        print(node.data)
        for obj in node.childs:
            self._printTree(obj)

    def getRules(self):
        reglas = []
        if self.root is not None:
            self.getRulesAux(self.root, [], reglas)
        for obj in reglas:
            print(obj)
        return reglas

    def getRulesAux(self, node, condiciones, reglas):
        if node.esHoja():
            reglas.append(' AND '.join(condiciones) + (' -> ' if condiciones else '-> ') + node.info())
            return
        for i in range(len(node.childs)):
            condicion = node.data['name'] + ' ' + node.data['labels'][i]
            self.getRulesAux(node.childs[i], condiciones + [condicion], reglas)

    def cantidadHojas(self):
        if self.root is None:
            return 0
        return self.cantidadHojasAux(self.root)

    def cantidadHojasAux(self, nodo):
        if nodo.esHoja():
            return 1
        return sum(self.cantidadHojasAux(obj) for obj in nodo.childs)

    def altura(self):
        if self.root is None:
            return 0
        return self.alturaAux(self.root)

    def alturaAux(self, nodo):
        if nodo.esHoja():
            return 0
        return 1 + max(self.alturaAux(hijo) for hijo in nodo.childs)
