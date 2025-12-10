"""
Construye matriz de similitud general a partir de varias matrices cuadradas (2475x2475)
usando un promedio ponderado.

Cada archivo H5 debe contener:
- dataset: "matriz"   (float64, cuadrada)
- dataset: "cvegeo"   (lista de municipios en el mismo orden)

CONFIGURA:
    ARCHIVOS_MATRICES
    PESOS
    RUTA_SALIDA
"""

import h5py
import pandas as pd
import numpy as np


# ------------------------------------------------------------
# 1. Leer matriz desde archivo H5
# ------------------------------------------------------------

def cargar_matriz(path):
    """
    Devuelve DataFrame con filas y columnas = cvegeo.
    """
    with h5py.File(path, "r") as f:
        matriz = f["matriz"][:]                # (2475 x 2475)
        cvegeo = [c.decode("utf-8") for c in f["cvegeo"][:]]

    df = pd.DataFrame(matriz, index=cvegeo, columns=cvegeo)
    return df


# ------------------------------------------------------------
# 2. Alinear matrices al mismo orden (por si acaso)
# ------------------------------------------------------------

def alinear_matriz(df, orden):
    """
    Reordena filas y columnas según `orden`.
    """
    return df.loc[orden, orden]


# ------------------------------------------------------------
# 3. Construir matriz general con pesos
# ------------------------------------------------------------

def construir_matriz_general(matrices, pesos):
    """
    matrices: lista de DataFrames cuadradas
    pesos: lista de floats (suma = 1)
    """
    assert len(matrices) == len(pesos), "Debe haber un peso por matriz."

    # Asegurar suma 1
    suma = sum(pesos)
    pesos = [p / suma for p in pesos]

    matriz_final = np.zeros_like(matrices[0].values)

    for df, w in zip(matrices, pesos):
        matriz_final += w * df.values

    # Convertir nuevamente a DataFrame
    cvegeo = matrices[0].index.tolist()
    return pd.DataFrame(matriz_final, index=cvegeo, columns=cvegeo)


# ------------------------------------------------------------
# 4. Guardar resultado en H5
# ------------------------------------------------------------

def guardar_h5(df, path_salida):
    cvegeo = df.index.astype(str).tolist()
    arr_cvegeo = np.array(cvegeo, dtype="S5")  # guardar como bytes
    matriz = df.values.astype("float64")

    with h5py.File(path_salida, "w") as f:
        f.create_dataset("cvegeo", data=arr_cvegeo)
        f.create_dataset("matriz", data=matriz)

    print(f"[OK] Matriz guardada en: {path_salida}")


# ------------------------------------------------------------
# 5. MAIN
# ------------------------------------------------------------

if __name__ == "__main__":

    # ================== CONFIGURA AQUÍ =======================
    ARCHIVOS_MATRICES = [
        "outputs/matriz_categorica.h5",
        "outputs/matriz_numerica.h5",
        "outputs/matriz_precipitacion.h5",
        "outputs/matriz_sequia.h5"
    ]

    PESOS = [0.25, 0.25, 0.25, 0.25]   # asignar peso a cada matriz

    RUTA_SALIDA = "outputs/matriz_sim_general.h5"
    # ===========================================================

    print("[INFO] Cargando matrices...")

    matrices = [cargar_matriz(p) for p in ARCHIVOS_MATRICES]

    # Asegurar mismo orden
    orden = matrices[0].index.tolist()
    matrices = [alinear_matriz(df, orden) for df in matrices]

    print("[INFO] Construyendo matriz general...")
    matriz_general = construir_matriz_general(matrices, PESOS)

    print("[INFO] Guardando archivo...")
    guardar_h5(matriz_general, RUTA_SALIDA)

    print("[DONE] Matriz general construida exitosamente.")
