import os
import numpy as np


class util:

    @staticmethod
    def cleanTempFolder(folder):
        """Crea la carpeta si no existe y borra los archivos que contiene."""
        os.makedirs(folder, exist_ok=True)
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)

    @staticmethod
    def entropy(counts):
        """Entropia (base 2) a lo largo del ultimo eje de `counts` (conteos por clase)."""
        counts = np.asarray(counts, dtype=float)
        total = counts.sum(axis=-1)
        with np.errstate(divide='ignore', invalid='ignore'):
            p = counts / np.expand_dims(total, -1)
            terms = np.where(counts > 0, p * np.log2(p), 0.0)
        return -terms.sum(axis=-1)

    @staticmethod
    def fatherEntropy(counts):
        return float(util.entropy(counts))

    @staticmethod
    def splitInfo(sizes):
        """Split info de una particion cuyas ramas tienen `sizes` instancias."""
        return float(util.entropy(sizes))

    @staticmethod
    def gainRatio(gain, split_info):
        if split_info <= 0:
            return 0.0
        return gain / split_info
