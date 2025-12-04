# src/tda/tda_matriz.py
import os
import numpy as np
import h5py
from joblib import Parallel, delayed
from src.utils.tda_utils import (
    normalizar_serie,
    takens_embedding,
    obtener_diagrama,
    distancia_tda, distancia_diagramas
)
from math import ceil
from multiprocessing import cpu_count
from src.utils.tda_utils import distancia_diagramas  # función que calcula dist entre dos diagramas

def construir_matriz_tda_from_cache(cache_h5, salida_h5, n_jobs=None, block_size=200):
    with h5py.File(cache_h5, "r") as f:
        cves = [c.decode() if isinstance(c, bytes) else c for c in f["cvegeo"][:]]
        grp = f["diagramas"]

        # cargar diagramas en memoria (lista) -- si pesan mucho, se pueden procesar por bloques
        diagramas = []
        for c in cves:
            raw = grp[c]
            dgm = pickle.loads(raw[()])
            diagramas.append(dgm)

    n = len(cves)
    os.makedirs(os.path.dirname(salida_h5), exist_ok=True)

    # Escribir en HDF5 por bloques
    with h5py.File(salida_h5, "w") as f_out:
        dset = f_out.create_dataset("distancias", shape=(n, n), dtype=np.float32, compression="gzip")
        f_out.create_dataset("cvegeo", data=np.array(cves, dtype='S5'))

        if n_jobs is None:
            n_jobs = max(1, cpu_count() - 1)

        def procesar_i(i):
            fila = np.zeros(n, dtype=np.float32)
            dgi = diagramas[i]
            for j in range(i, n):
                fila[j] = distancia_diagramas(dgi, diagramas[j])
            return i, fila

        # Parallelizar por filas (o por bloques si n muy grande)
        results = Parallel(n_jobs=n_jobs)(delayed(procesar_i)(i) for i in range(n))

        for i, fila in results:
            dset[i, i:] = fila[i:]
            dset[i:, i] = fila[i:]

        # --- Reflejo inferior ---
        print("Reflejando matriz...")
        for i in range(n):
            for j in range(i + 1, n):
                d_matriz[j, i] = d_matriz[i, j]

    print(f"Matriz TDA guardada en: {salida_h5}")
