import os
import numpy as np
import pandas as pd
import h5py
from math import ceil
from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import MinMaxScaler

# -------------------------------------------------------
# helper: pivotar si el DF está en formato largo (CVEGEO, valid_time, value)
# -------------------------------------------------------
def pivot_if_long(df):
    # si tiene columna 'valid_time' (o tipo datetime) y más de 2 columnas -> es formato largo
    cols = set(df.columns.str.lower())
    if "valid_time" in df.columns or "valid_time" in cols:
        # intentar detectar la columna de valor (la que no sea CVEGEO o valid_time)
        possible = [c for c in df.columns if c not in ("CVEGEO", "valid_time")]
        if len(possible) == 0:
            raise ValueError("Formato largo detectado pero no hay columna de valor.")
        val_col = possible[0]
        # crear columna mes (YYYY_MM) si es mensual (o fecha si ya lo es)
        df["valid_time"] = pd.to_datetime(df["valid_time"], errors="coerce")
        # si el dataset tiene hora, convertimos a fecha (diario) o a mes si queremos mensual
        # aquí asumimos que para matrices mensuales queremos agrupar por mes -> YYYY_MM
        df["mes"] = df["valid_time"].dt.to_period("M").astype(str)
        pivot = df.pivot_table(index="CVEGEO", columns="mes", values=val_col, aggfunc="mean")
        # devolver pivot como dataframe con columnas ordenadas cronológicamente
        pivot = pivot.reindex(sorted(pivot.columns), axis=1).reset_index()
        return pivot
    else:
        # si ya está ancho (CVEGEO + columnas mes), devolver tal cual
        return df

# ======================================================
# FUNCIÓN QUE CALCULA UN BLOQUE DEL TRIÁNGULO SUPERIOR
# ======================================================
def calcular_bloque(args):
    data, start_i, end_i = args
    n = data.shape[0]

    bloque = []
    for i in range(start_i, end_i):
        fila = np.zeros(n, dtype=np.float32)
        vi = data[i]
        # calcular j desde i hasta n-1
        for j in range(i, n):
            dist = np.linalg.norm(vi - data[j])
            fila[j] = 1.0 / (1.0 + dist)   # similitud 1/(1+d)
        bloque.append((i, fila))
    return bloque

# ======================================================
# FUNCIÓN GENERAL PARA CONSTRUIR MATRIZ OPTIMIZADA (AHORA ROBUSTA)
# ======================================================
def construir_matriz_optimizada(path_parquet, salida_h5, normalizar=False, modo="mensual"):
    """
    path_parquet: ruta a parquet que puede estar en formato ancho (CVEGEO + meses) o largo (CVEGEO, valid_time, value).
    salida_h5: ruta de salida .h5
    normalizar: aplicar MinMax por columna (meses)
    modo: 'mensual' (por defecto) - usado para decidir agrupación al pivotar
    """
    # crear carpeta de salida si no existe
    os.makedirs(os.path.dirname(salida_h5) or ".", exist_ok=True)

    df = pd.read_parquet(path_parquet)

    # si viene en formato largo, pivotar a ancho
    df = pivot_if_long(df)

    # asegurar que CVEGEO exista y esté ordenado
    if "CVEGEO" not in df.columns:
        raise ValueError("Después del pivot falta columna 'CVEGEO'")

    df = df.sort_values("CVEGEO").reset_index(drop=True)

    municipios = df["CVEGEO"].astype(str).to_numpy()
    # quitar la columna CVEGEO y convertir a numpy (valores float)
    values = df.drop(columns=["CVEGEO"]).to_numpy(dtype=np.float32)

    print("\n=== Construcción optimizada ===")
    print(f"Municipios: {values.shape[0]}, Meses: {values.shape[1]}")
    print(f"Normalizar: {normalizar}")

    X = values
    if normalizar:
        scaler = MinMaxScaler()
        # aplicar columna por columna (fit_transform sobre filas)
        X = scaler.fit_transform(X)

    n = X.shape[0]

    # Crear archivo HDF5 y dataset
    with h5py.File(salida_h5, "w") as h5:
        d_matriz = h5.create_dataset(
            "matriz",
            shape=(n, n),
            dtype=np.float32,
            compression="gzip"
        )
        # guardar cvegeo como bytes fixed-length
        h5.create_dataset("cvegeo", data=municipios.astype("S5"), compression="gzip")

        # paralelizar por bloques
        num_workers = max(1, min(cpu_count() - 1,  max(1, n)))  # no pedir > n
        bloque_tamano = ceil(n / max(1, num_workers))

        print(f"Usando {num_workers} procesos, bloque: {bloque_tamano}")

        argumentos = []
        for k in range(num_workers):
            i0 = k * bloque_tamano
            i1 = min((k + 1) * bloque_tamano, n)
            if i0 >= i1:
                continue
            argumentos.append((X, i0, i1))

        # pool de procesos
        with Pool(num_workers) as pool:
            for bloque in pool.imap(calcular_bloque, argumentos):
                for (i, fila_sup) in bloque:
                    # longitud del triángulo en esta fila
                    long = n - i
                    # fila_sup tiene tamaño n, pero solo nos interesan las posiciones i..n-1
                    d_matriz[i, i:i+long] = fila_sup[i:i+long]
                    d_matriz[i:i+long, i] = fila_sup[i:i+long]  # reflejo

    print(f"Matriz guardada en: {salida_h5}")
    print("OK ✔️\n")



