import numpy as np
import pandas as pd
import h5py
from math import ceil
from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import MinMaxScaler


# ======================================================
# FUNCIÓN QUE CALCULA UN BLOQUE DEL TRIÁNGULO SUPERIOR
# ======================================================
def calcular_bloque(args):
    data, start_i, end_i = args
    n = data.shape[0]

    bloque = []
    for i in range(start_i, end_i):
        fila = np.zeros(n, dtype=np.float32)
        for j in range(i, n):  # triángulo superior
            dist = np.linalg.norm(data[i] - data[j])
            fila[j] = 1 / (1 + dist)
        bloque.append((i, fila))

    return bloque


# ======================================================
# FUNCIÓN GENERAL PARA CONSTRUIR MATRIZ OPTIMIZADA
# ======================================================
def construir_matriz_optimizada(path_parquet, salida_h5, normalizar=False):

    df = pd.read_parquet(path_parquet)

    if "CVEGEO" in df.columns:
        df = df.set_index("CVEGEO")

    municipios = df.index.to_numpy()
    X = df.to_numpy(dtype=np.float32)

    print("\n=== Construcción optimizada ===")
    print(f"Municipios: {X.shape[0]}, Meses: {X.shape[1]}")
    print(f"Normalizar: {normalizar}")

    if normalizar:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

    n = X.shape[0]

    # Crear archivo HDF5
    with h5py.File(salida_h5, "w") as h5:
        d_matriz = h5.create_dataset(
            "matriz",
            shape=(n, n),
            dtype=np.float32,
            compression="gzip"
        )
        h5.create_dataset("cvegeo", data=municipios.astype("S"), compression="gzip")

        # paralelizar por bloques
        num_workers = max(2, cpu_count() - 1)
        bloque_tamano = ceil(n / num_workers)

        print(f"Usando {num_workers} procesos, bloque: {bloque_tamano}")

        argumentos = []
        for k in range(num_workers):
            i0 = k * bloque_tamano
            i1 = min((k + 1) * bloque_tamano, n)
            argumentos.append((X, i0, i1))

        with Pool(num_workers) as pool:
            for bloque in pool.imap(calcular_bloque, argumentos):
                for (i, fila_sup) in bloque:
                    long = n-i
                    d_matriz[i, i:i+long] = fila_sup[i:i+long]
                    d_matriz[i:i+long, i] = fila_sup[i:i+long]  # reflejo

    print(f"Matriz guardada en: {salida_h5}")
    print("OK ✔️\n")


# ======================================================
# WRAPPERS PARA RADIACIÓN Y SEQUÍA
# ======================================================
def matriz_radiacion():
    construir_matriz_optimizada(
        path_parquet="data/series_tiempo/Radiacion_municipal_limpia.parquet",
        salida_h5="data/matrices_mensuales/matriz_radiacion.h5",
        normalizar=True
    )


def matriz_sequia():
    construir_matriz_optimizada(
        path_parquet="data/series_tiempo/Sequia_municipal_limpia.parquet",
        salida_h5="data/matrices_mensuales/matriz_sequia.h5",
        normalizar=False
    )



