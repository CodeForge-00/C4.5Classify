# C4.5 desde cero: clasificación de la calidad del vino tinto

Implementación didáctica del algoritmo de árboles de decisión **C4.5** (Quinlan, 1993) en Python, sin
librerías de machine learning. Se aplica al dataset *Wine Quality (red)* de UCI para predecir la
calidad de un vino (3 a 8) a partir de 11 medidas fisicoquímicas.

Este documento explica **qué hace el algoritmo, cómo está implementado y cómo usarlo**, para que
cualquiera pueda entenderlo sin haber visto el código antes.

## Contenido

1. [Inicio rápido](#1-inicio-rápido)
2. [Estructura del proyecto](#2-estructura-del-proyecto)
3. [El problema y los datos](#3-el-problema-y-los-datos)
4. [Cómo funciona un árbol de decisión](#4-cómo-funciona-un-árbol-de-decisión)
5. [Construcción del árbol (C4.5)](#5-construcción-del-árbol-c45)
6. [Poda](#6-poda)
7. [Predicción y evaluación](#7-predicción-y-evaluación)
8. [Estructuras de datos](#8-estructuras-de-datos)
9. [Qué implementa y qué no frente al C4.5 original](#9-qué-implementa-y-qué-no-frente-al-c45-original)
10. [Resultados de referencia](#10-resultados-de-referencia)
11. [Cómo extenderlo](#11-cómo-extenderlo)
12. [Referencias](#12-referencias)

---

## 1. Inicio rápido

**Requisitos:** Python 3.6 o superior, `numpy` y `pandas` (probado con Python 3.9 y pandas 2.3).
Los scripts de análisis previo (sección 2) necesitan además `openpyxl`, `scikit-learn` y `matplotlib`.

```bash
pip install numpy pandas            # lo mínimo para el clasificador
python main.py                      # validación cruzada de 10 folds con poda pesimista
```

Opciones de `main.py`:

| Opción | Valores | Por defecto | Qué hace |
|---|---|---|---|
| `--pruning` | `pessimistic`, `penalty`, `both` | `pessimistic` | Variante de poda. `both` corre las dos sobre los mismos folds y muestra una tabla comparativa. |
| `--cf` | número en (0, 0.5) | `0.25` | Nivel de confianza de la poda pesimista. Más bajo = poda más. |
| `--omega` | número ≥ 0 | `1.0` | Costo por hoja de la poda por penalidad. Más alto = poda más. |
| `--k` | entero ≥ 2 | `10` | Número de folds. |
| `--seed` | entero | aleatorio | Semilla para repetir exactamente el reparto de folds. |

Ejemplos:

```bash
python main.py --pruning both --seed 0           # compara las dos podas con los mismos folds
python main.py --pruning penalty --omega 3       # poda por penalidad más agresiva
python main.py --pruning pessimistic --cf 0.10   # poda pesimista más agresiva
```

Usarlo desde código:

```python
from c45 import C45
from training_set import Subsets

folds = Subsets(seed=0).kFoldCrossValidation(k=10)
train, test = folds[0]['training'], folds[0]['testing']   # DataFrames; la última columna es la clase

modelo = C45(pruning='pessimistic', cf=0.25).fit(train, classes=[3, 4, 5, 6, 7, 8])
modelo.postPoda()                                   # sin esta llamada el árbol queda sin podar
modelo.getRules()                                   # imprime las reglas del árbol
error, confusion = modelo.validationError(test)     # error en prueba y matriz de confusión
predicciones = modelo.predict(test)                 # lista con la clase predicha de cada fila
```

> Ejecuta los scripts desde esta carpeta con `python main.py` (no con `python -I`): los módulos se
> importan entre sí por nombre y necesitan que la carpeta del script esté en el path de Python.

---

## 2. Estructura del proyecto

| Archivo | Responsabilidad |
|---|---|
| `main.py` | Punto de entrada. Reparte los datos en folds, entrena un árbol por fold, lo poda, lo evalúa y promedia los resultados. |
| `c45.py` | Clase `C45`: construcción del árbol, poda, predicción y métricas de error. Es el corazón del proyecto. |
| `cutpoint.py` | Clase `cutpoint`: encuentra el mejor umbral `<= t` / `> t` para un atributo numérico. |
| `pruning.py` | Variantes de poda: `PessimisticPruning` y `PenaltyPruning` (comparten la interfaz `Pruning`). |
| `Tree.py` | Estructura de datos `Tree` / `TreeNode`: conteo de hojas, altura y generación de reglas. |
| `util.py` | Funciones matemáticas: entropía, *split info*, *gain ratio*. |
| `training_set.py` | Clase `Subsets`: validación cruzada estratificada (`kFoldCrossValidation`) y hold-out repetido (`RandomSubSampling`). |
| `correlation_matrix.py`, `covariance_matrix.py`, `principal_component_analysis.py` | Análisis exploratorio independiente del clasificador (correlación, covarianza y PCA con *scree plot*). Escriben en `Data/Preprocessing/`. |
| `Data/` | `winequality-red.csv` (el que usa el clasificador), su copia en `.xlsx` (la usan los scripts de análisis), `winequality.names` (descripción de UCI) y `NOTA` (enlace de origen). |

Dependencias entre módulos: `main` → `c45`, `pruning`, `training_set`; `c45` → `Tree`, `cutpoint`,
`pruning`; `cutpoint` → `util`.

---

## 3. El problema y los datos

- **Origen:** <https://archive.ics.uci.edu/ml/datasets/Wine+Quality> (vino tinto *vinho verde*, Cortez et al., 2009).
- **Instancias:** 1599. **Atributos:** 11, todos numéricos y continuos: acidez fija, acidez volátil,
  ácido cítrico, azúcar residual, cloruros, dióxido de azufre libre y total, densidad, pH, sulfatos y alcohol.
- **Clase (`quality`):** entero de 3 a 8. No hay valores faltantes.
- **Distribución de clases (muy desbalanceada):**

| Calidad | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|
| Instancias | 10 | 53 | 681 | 638 | 199 | 18 |

Las calidades 5 y 6 son el 82 % de los datos. Consecuencia práctica: un clasificador acierta bastante
en 5 y 6 y casi nunca en 3, 4 y 8 (se ve en la matriz de confusión). La *exactitud* global por sí sola
engaña; mire siempre la matriz de confusión.

El formato del CSV es una fila de cabecera con los nombres de columna y la **clase siempre en la última
columna**. Cualquier otro CSV numérico con esa forma funciona sin cambios en el código.

---

## 4. Cómo funciona un árbol de decisión

Un árbol de decisión es una serie de preguntas encadenadas. Cada **nodo interno** hace una pregunta sobre
un atributo (por ejemplo «¿alcohol ≤ 10.1?») y dirige la instancia a una de sus ramas. Cada **hoja**
guarda una clase. Para clasificar un vino se parte de la raíz, se responden las preguntas y se devuelve
la clase de la hoja a la que se llega.

```
                    alcohol <= 10.1 ?
                   /                 \
                 sí                   no
                /                       \
   sulphates <= 0.54 ?                  clase 6
        /        \
      sí          no
      |            |
   clase 5      clase 6
```

El aprendizaje consiste en **decidir qué pregunta hacer en cada nodo**. La idea de C4.5 es hacer la
pregunta que más «ordene» los datos, es decir, la que deja los grupos resultantes más homogéneos en clase.
Para medirlo se usa la entropía.

---

## 5. Construcción del árbol (C4.5)

El árbol se construye de forma **recursiva**, empezando con todos los datos de entrenamiento en la raíz
(método `C45._build` en `c45.py`):

```
construir(datos):
    si los datos son de una sola clase, o son muy pocos, o ninguna pregunta ayuda:
        devolver una hoja con la clase mayoritaria
    elegir la mejor pregunta (atributo + umbral)
    dividir los datos en  "atributo <= umbral"  y  "atributo > umbral"
    nodo.hijo_izquierdo = construir(datos de la izquierda)
    nodo.hijo_derecho   = construir(datos de la derecha)
```

### 5.1 Entropía

Mide el desorden (la mezcla de clases) de un conjunto. Si `p_i` es la proporción de instancias de la clase `i`:

```
Entropía(S) = - Σ p_i · log2(p_i)
```

- Un conjunto de una sola clase tiene entropía 0 (orden total).
- Dos clases al 50 % tienen entropía 1 (máximo desorden con dos clases).

Implementada en `util.entropy`.

### 5.2 Ganancia de información

Cuánto baja la entropía al dividir un conjunto `S` en subconjuntos `S_j` con una pregunta:

```
Ganancia = Entropía(S) - Σ (|S_j| / |S|) · Entropía(S_j)
```

Una ganancia alta significa que la pregunta separa bien las clases.

### 5.3 Atributos continuos: búsqueda del umbral

Los 11 atributos son números reales, así que la pregunta es de la forma `atributo <= t`. El mejor `t` se
busca **en cada nodo, con los datos que llegaron a ese nodo** (`cutpoint.bestCut`):

1. Se ordenan las instancias por el valor del atributo.
2. Cada par de valores distintos consecutivos define un umbral candidato. Se usa como umbral el valor
   menor del par (así el umbral es siempre un valor real del dato).
3. Con sumas acumuladas por clase se calcula la ganancia de todos los candidatos de una sola vez.
4. Se elige el candidato de mayor ganancia.

Dos detalles de C4.5 que se aplican aquí:

- **Mínimo de instancias por rama (`min_objs`, por defecto 2):** no se aceptan umbrales que dejen menos
  de `min_objs` instancias a algún lado. Evita hojas de una sola instancia.
- **Corrección de Quinlan (1996):** a la ganancia se le resta `log2(nº de umbrales candidatos) / N`.
  Compensa que un atributo con muchos valores distintos tiene más oportunidades de «tener suerte» al azar.
  Si tras la corrección la ganancia no es positiva, el atributo no se usa en ese nodo.

Como el umbral se recalcula en cada nodo, **un mismo atributo puede usarse varias veces en un camino**
con umbrales distintos (por ejemplo `alcohol <= 10.1` arriba y `alcohol <= 9.1` más abajo).

#### Ejemplo con números

Diez vinos, un solo atributo (`alcohol`) y dos clases (5 y 6):

| alcohol | 9.0 | 9.2 | 9.5 | 10.0 | 10.2 | 10.5 | 11.0 | 11.5 | 12.0 | 12.5 |
|---|---|---|---|---|---|---|---|---|---|---|
| calidad | 5 | 5 | 5 | 5 | 6 | 5 | 6 | 6 | 6 | 6 |

- Conjunto completo: 5 vinos de cada clase → `Entropía = 1.000`.
- Umbral `alcohol <= 10.0`: izquierda con 4 vinos, todos de calidad 5 (entropía 0); derecha con 6 vinos,
  1 de calidad 5 y 5 de calidad 6 (entropía 0.650).
- `Ganancia = 1.000 − (0.4·0 + 0.6·0.650) = 0.610`.
- Probando todos los umbrales, `<= 10.0` es el de mayor ganancia (el siguiente candidato, `<= 10.2`, da 0.278).

`cutpoint.bestCut` devuelve para este caso `threshold = 10.0`. El valor de `gain` que reporta es 0.293 y
no 0.610 porque ya incluye la corrección de Quinlan (`log2(9)/10 = 0.317`).

### 5.4 Gain ratio y elección del atributo

La ganancia favorece preguntas que dividen en ramas muy desiguales o muy numerosas. Para corregirlo C4.5
divide entre la *split info*, la entropía del propio reparto de instancias entre las ramas:

```
SplitInfo  = - Σ (|S_j| / |S|) · log2(|S_j| / |S|)
GainRatio  = Ganancia / SplitInfo
```

En el ejemplo: ramas de 4 y 6 instancias → `SplitInfo = 0.971`, y `GainRatio = 0.610 / 0.971 = 0.628` (sin la corrección de Quinlan).

El gain ratio solo puede sobrevalorar divisiones muy desbalanceadas con ganancia casi nula. Por eso C4.5
aplica una regla en dos pasos (`C45._chooseSplit`):

1. Para cada atributo se calcula su mejor umbral y su ganancia.
2. Se descartan los atributos con ganancia **por debajo del promedio**, y entre los restantes se elige el de **mayor gain ratio**.

### 5.5 Cuándo un nodo se convierte en hoja

Un nodo no se divide (y se convierte en hoja con la clase más frecuente) cuando:

- todas sus instancias son de la misma clase, o
- tiene menos de `2 · min_objs` instancias, o
- ningún atributo ofrece un umbral válido con ganancia positiva.

En empates de clase gana la primera en orden creciente.

---

## 6. Poda

Si se deja crecer el árbol hasta el final aprende hasta el ruido de los datos de entrenamiento
(**sobreajuste**): el error de entrenamiento baja mucho pero el de datos nuevos no. La **post-poda**
construye primero el árbol completo y luego lo recorta.

El método usado es el **reemplazo de subárboles** (`C45.postPoda`): se recorre el árbol de abajo hacia
arriba y, en cada nodo interno, se compara

- el error del **subárbol** actual (suma de los errores de sus hojas), contra
- el error que tendría el nodo **si fuera una hoja** con su clase mayoritaria.

Si convertirlo en hoja no empeora el criterio, el subárbol se reemplaza por una hoja. Gracias a que cada
nodo guarda la distribución de clases de las instancias que le llegaron (`resumen_clases`), la decisión
se toma sin volver a mirar los datos.

Hay dos criterios para decidir, y se escoge con `pruning=` (código en `pruning.py`):

### 6.1 Poda pesimista de C4.5 (`PessimisticPruning`, por defecto)

El error observado en el entrenamiento es demasiado optimista (un árbol siempre acierta más con los
datos que usó para aprender). C4.5 lo corrige **sumando un error extra** a cada hoja: el límite superior
de un intervalo de confianza del error binomial, con nivel de confianza `cf` (0.25 por defecto).

```
error_pesimista(hoja) = errores_observados + AddErrs(N, errores_observados)
```

donde `N` es el número de instancias de la hoja. `AddErrs` (de Quinlan) crece cuando hay pocas instancias:
una hoja pequeña es poco fiable, así que se le asigna más error.

Ejemplos reales con `cf = 0.25`:

- Una hoja con 5 instancias y 0 errores se estima en **1.21** errores, no en 0.
- Un nodo con 14 instancias (8 de una clase y 6 de otra) que se divide en dos hojas puras: como hoja valdría
  `6 + AddErrs(14, 6) = 7.75`; sus dos hojas suman `2.51`. Como `7.75 > 2.51 + 0.1`, **no se poda**.
- Un nodo con 14 instancias y 3 errores, dividido en hojas de 8 y 6 instancias con 1 error cada una:
  como hoja `4.69`, sus hojas suman `4.67`. Como `4.69 ≤ 4.67 + 0.1`, **se poda**: la división no aporta lo suficiente.

**Regla de poda:** `error_pesimista(nodo como hoja) ≤ Σ error_pesimista(hojas del subárbol) + 0.1`
(el `0.1` favorece el árbol más simple en empates, igual que C4.5).

`cf` más bajo → intervalos más anchos → más error extra a las hojas pequeñas → **poda más**.

Error de generalización reportado con esta variante: `Σ error_pesimista(hojas) / N`.

### 6.2 Poda por penalidad (`PenaltyPruning`)

Criterio más simple: se penaliza cada hoja con un costo fijo `omega` (complejidad del modelo).

```
error_generalización = error_entrenamiento + omega · nº_hojas / N
```

Un nodo se poda si eso no empeora:

```
errores_nodo_como_hoja + omega  ≤  errores_subárbol + omega · hojas_subárbol
```

Con el mismo nodo de 14 instancias dividido en dos hojas puras (6 errores como hoja, 0 en el subárbol): se
poda solo si `6 + omega ≤ 0 + 2·omega`, es decir, **`omega ≥ 6`**. Con `omega = 1` el árbol se queda con su división.

`omega` más alto → **poda más**.

### 6.3 Cuál usar

| | Pesimista | Penalidad |
|---|---|---|
| Origen | Es la poda original de C4.5 | Criterio propio, más simple |
| Parámetro | `cf` (0.25) | `omega` (1.0) |
| Fundamento | Estadístico (límite de confianza binomial) | Heurístico (costo por hoja) |
| Aspecto práctico | Poda poco con datos ruidosos como estos | Su agresividad se ajusta directamente con `omega` |

Ninguna de las dos es «la correcta» para todos los datos: con ruido alto como el de este dataset se obtiene
mejor error de validación podando más (ver sección 10). Lo recomendable es elegir `cf` u `omega` por
validación **dentro del conjunto de entrenamiento**, no mirando el error de prueba.

---

## 7. Predicción y evaluación

**Predicción (`C45.predictClass`):** se recorre el árbol desde la raíz; en cada nodo se compara el valor
del atributo con el umbral (`<=` va a la rama izquierda, `>` a la derecha) hasta llegar a una hoja, cuya
clase es la predicción. No hay ramas faltantes: toda división crea siempre las dos ramas.

**Validación cruzada estratificada (`Subsets.kFoldCrossValidation`):**

1. Los datos se reparten en `k` folds de tamaño casi igual (160, ..., 159 con `k = 10`).
2. Se hace por clase: las instancias de cada clase se mezclan al azar y se reparten por turnos entre
   los folds. Así cada fold conserva la proporción de clases del conjunto completo.
3. Para cada fold se entrena con los otros `k − 1` y se evalúa con el fold reservado. Ningún dato de
   prueba interviene en el entrenamiento ni en la poda.

**Métricas** (`main.py` imprime el promedio de los `k` folds):

| Métrica | Significado |
|---|---|
| `training_error` | Fracción de instancias de entrenamiento mal clasificadas por el árbol podado. |
| `generalization_error` | Estimación del error futuro que da el criterio de poda (6.1 o 6.2). No usa los datos de prueba. |
| `validation_error` | Fracción de instancias del fold de prueba mal clasificadas. **Es la métrica que importa.** |
| `hojas_prepoda`, `hojas_postpoda` | Número de hojas antes y después de podar (tamaño del modelo). |
| `altura` | Longitud del camino más largo de la raíz a una hoja. |

**Matriz de confusión:** filas = clase real, columnas = clase predicha, ordenadas por calidad
(3, 4, 5, 6, 7, 8). La diagonal son los aciertos. `main.py` imprime la de cada fold y la suma de todos.

---

## 8. Estructuras de datos

Cada nodo del árbol (`TreeNode`) tiene una lista `childs` y un diccionario `data`.

**Hoja** (`childs` vacío):

| Clave | Contenido |
|---|---|
| `tag_name` | Clase predicha (la más frecuente entre sus instancias). |
| `classify_error` | `1 − (instancias de la clase mayoritaria / instancias de la hoja)`. |
| `resumen_clases` | Lista `[{'name': clase, 'count': n}, ...]` con las instancias de entrenamiento de cada clase. |

**Nodo interno**: las mismas tres claves (para poder podarlo) más:

| Clave | Contenido |
|---|---|
| `name` | Nombre del atributo por el que se divide. |
| `index` | Posición de ese atributo en los datos. |
| `threshold` | Umbral `t`. |
| `labels` | `['<=t', '>t']`, texto de cada rama; `childs[0]` es la rama `<=`, `childs[1]` la rama `>`. |

Las reglas (`Tree.getRules`) se generan recorriendo cada camino de la raíz a una hoja:

```
alcohol <=10.1 AND free sulfur dioxide <=33.0 AND sulphates <=0.54 -> 5 (Classify Error: 0.0)
```

---

## 9. Qué implementa y qué no frente al C4.5 original

**Implementado:** ganancia de información y gain ratio; selección entre atributos con ganancia superior al
promedio; búsqueda de umbral por nodo para atributos continuos con la corrección de Quinlan (1996);
mínimo de instancias por rama; poda pesimista por reemplazo de subárboles; poda alternativa por penalidad.

**No implementado (limitaciones conocidas):**

- **Atributos categóricos:** solo hay atributos numéricos. Las divisiones son siempre binarias (`<=` / `>`).
- **Valores faltantes:** el código asume que no hay (el dataset no tiene).
- **Subtree raising:** la poda solo reemplaza un subárbol por una hoja; no sube una rama hija a la posición del padre.
- **Poda de reglas:** C4.5 también puede convertir el árbol en reglas y podarlas por separado (C4.5rules); aquí no.
- **Pesos por instancia, costos de error por clase, *boosting*:** no.
- **Escalabilidad:** el código está pensado para legibilidad y para datasets de miles de filas, no para millones.

Además, `AddErrs` se aplica con la fórmula del código de Quinlan (aproximación normal): difiere hasta
unos 0.1 errores del límite binomial exacto en hojas pequeñas.

---

## 10. Resultados de referencia

10-fold estratificado con `--seed 0` (los números cambian con la semilla):

| Promedio de los 10 folds | Poda pesimista (`cf = 0.25`) | Poda por penalidad (`omega = 1`) |
|---|---|---|
| Error de entrenamiento | 0.102 | 0.126 |
| Error de generalización (estimado) | 0.269 | 0.241 |
| **Error de validación** | **0.399** | **0.399** |
| Hojas antes de podar | 264.8 | 264.8 |
| Hojas después de podar | 199.5 | 166.0 |
| Altura | 19.3 | 19.0 |

Interpretación: acierta alrededor del 60 % de los vinos. Las dos podas por defecto dejan árboles muy
grandes (más de 160 hojas) con un error de entrenamiento muy por debajo del de validación: hay sobreajuste
y margen para podar más. Podar más (por ejemplo `--pruning penalty --omega 3`, unas 40 hojas) deja el error
de validación en el mismo rango (≈ 0.39 a 0.40), con un árbol mucho más simple.

**Precauciones al comparar resultados:**

- El error de validación depende del reparto de folds: con otra semilla o con otro método de reparto cambia
  varios puntos. Compare variantes **siempre con los mismos folds** (`--seed`).
- Elegir `cf` u `omega` mirando el error de prueba de esos mismos folds da una cifra optimista.
- Las clases 3, 4 y 8 casi nunca se predicen bien: tienen muy pocas instancias.

---

## 11. Cómo extenderlo

**Añadir una variante de poda:** crear en `pruning.py` una subclase de `Pruning` e implementar tres métodos:

```python
class MiPoda(Pruning):
    name = 'mi_poda'

    def nodeError(self, n, e):
        """Error de un nodo con n instancias y e mal clasificadas (usado para hojas y para el nodo como hoja)."""
        return e

    def shouldPrune(self, leaf_error, subtree_error, subtree_leaves):
        """True si el nodo debe convertirse en hoja."""
        return leaf_error <= subtree_error

    def generalizationError(self, model):
        """Estimación del error de generalización del árbol completo."""
        return model.trainingError()
```

Después se usa con `C45(pruning=MiPoda())`. Si también quiere elegirla por nombre, añádala a `makePruning`.

**Otro dataset:** un CSV numérico con cabecera y la clase en la última columna. Cambie la ruta `data_file`
en `training_set.py`. Las etiquetas de clase deben ser números (enteros o decimales).

**Parámetros del árbol (`C45`):** `min_objs` (instancias mínimas por rama, por defecto 2), `pruning`, `cf`, `omega`.
Subir `min_objs` produce árboles más pequeños aun antes de podar.

---

## 12. Referencias

- J. R. Quinlan, *C4.5: Programs for Machine Learning*, Morgan Kaufmann, 1993. Algoritmo, gain ratio y poda pesimista.
- J. R. Quinlan, «Improved Use of Continuous Attributes in C4.5», *Journal of Artificial Intelligence Research* 4, 1996. Corrección de la ganancia para umbrales continuos.
- P. Cortez, A. Cerdeira, F. Almeida, T. Matos, J. Reis, «Modeling wine preferences by data mining from physicochemical properties», *Decision Support Systems* 47(4), 2009. Origen del dataset.
- P.-N. Tan, M. Steinbach, V. Kumar, *Introduction to Data Mining*. Entropía, ganancia, sobreajuste y el error de generalización con penalidad por hoja.
