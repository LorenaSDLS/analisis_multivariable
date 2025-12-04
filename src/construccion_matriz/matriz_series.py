import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from functools import partial


def similitud_series(seriesA, seriesB):
    """Similitud básica: correlación de Pearson, segura."""
    if np.all(np.isnan(seriesA)) or np.all(np.isnan(seriesB)):
        return 0.0
    corr = np.corrcoef(seriesA, seriesB)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(corr)


def tarea_comparacion(args):
    """Desempaqueta los argumentos. Compatible con ProcessPoolExecutor."""
    cveA, cveB, series_dict = args
    return cveA, cveB, similitud_series(series_dict[cveA], series_dict[cveB])


def construir_matriz_similitud(df):
    cves = df["CVEGEO"].tolist()
    series_dict = {
        cve: df.loc[df["CVEGEO"] == cve].iloc[:, 1:].values.flatten()
        for cve in cves
    }

    n = len(cves)
    matriz = np.zeros((n, n), dtype=np.float32)

    pares = [(i, j) for i, j in combinations(range(n), 2)]
    args = [(cves[i], cves[j], series_dict) for i, j in pares]

    print(f"Comparaciones totales: {len(args):,}")

    with ProcessPoolExecutor() as executor:
        for idx, (cveA, cveB, sim) in enumerate(executor.map(tarea_comparacion, args)):
            i = cves.index(cveA)
            j = cves.index(cveB)
            matriz[i, j] = matriz[j, i] = sim

            if idx % 50_000 == 0 and idx > 0:
                print(f"  {idx:,} comparaciones...")

    return cves, matriz
