# imputación basado en similitud

import pandas as pd
import numpy as np
import h5py


# ------------------------------------------------------------
# 1. OBTENER LISTA DE MUNICIPIOS SIMILARES (top-n)
# ------------------------------------------------------------

def obtener_similares(cvegeo, matriz_sim, top_n=5):
    if cvegeo not in matriz_sim.index:
        raise ValueError(f"{cvegeo} no está en la matriz de similitud")

    similares = matriz_sim.loc[cvegeo].drop(cvegeo)
    similares = similares.sort_values(ascending=False)

    return similares.head(top_n).index.tolist()



# ------------------------------------------------------------
# 2. OBTENER SERIES DE TIEMPO DE LOS VECINOS
# ------------------------------------------------------------

def obtener_series_vecinos(df_incompleto, vecinos):
    return df_incompleto[df_incompleto["CVEGEO"].isin(vecinos)].copy()



# ------------------------------------------------------------
# 3. PROMEDIAR SERIES DE TIEMPO
# ------------------------------------------------------------

def promediar_series(series_vecinos):
    tabla = series_vecinos.pivot_table(
        index="valid_time",
        values="variable",
        aggfunc="mean"
    ).sort_index()

    return tabla



# ------------------------------------------------------------
# 4. CREAR SERIE IMPUTADA PARA UN MUNICIPIO
# ------------------------------------------------------------

def imputar_serie(cvegeo_faltante, promedio_series):
    df = promedio_series.copy()
    df["CVEGEO"] = cvegeo_faltante
    df = df.reset_index()
    return df



# ------------------------------------------------------------
# 5. IMPUTAR UN MUNICIPIO
# ------------------------------------------------------------

def rellenar_un_municipio(cvegeo_faltante, matriz_sim, df_incompleto, top_n=5):
    vecinos = obtener_similares(cvegeo_faltante, matriz_sim, top_n)
    series_vecinos = obtener_series_vecinos(df_incompleto, vecinos)

    if series_vecinos.empty:
        raise ValueError(f"Vecinos de {cvegeo_faltante} no tienen series.")

    promedio = promediar_series(series_vecinos)
    serie_nueva = imputar_serie(cvegeo_faltante, promedio)

    return serie_nueva



# ------------------------------------------------------------
# 6. COMPLETAR TODO EL DATASET
# ------------------------------------------------------------

def completar_dataset(df_incompleto, matriz_sim, lista_cvegeo, top_n=5):
    df = df_incompleto.copy()

    cvegeo_existentes = set(df["CVEGEO"].unique())
    faltantes = [c for c in lista_cvegeo if c not in cvegeo_existentes]

    print(f"[INFO] Municipios totales: {len(lista_cvegeo)}")
    print(f"[INFO] Municipios con datos: {len(cvegeo_existentes)}")
    print(f"[INFO] Municipios faltantes: {len(faltantes)}")

    imputaciones = []

    for cve in faltantes:
        print(f"→ Imputando {cve} usando top-{top_n} similares...")
        nueva_serie = rellenar_un_municipio(cve, matriz_sim, df, top_n)
        imputaciones.append(nueva_serie)

    if imputaciones:
        df_completo = pd.concat([df] + imputaciones, ignore_index=True)
    else:
        df_completo = df

    print("[INFO] Dataset completado.")
    return df_completo



# ------------------------------------------------------------
# 7. MAIN
# ------------------------------------------------------------

if __name__ == "__main__":
    RUTA_LISTA_CVEGEO = "data/tabla_municipios.parquet"
    RUTA_DATASET_INCOMPLETO = "data/evap_diario.parquet"
    RUTA_MATRIZ_SIM = "outputs/matriz_completa/matriz_numerica.h5"
    RUTA_SALIDA = "data/evapCompleto.parquet"

    # Lista de municipios oficiales
    lista_cvegeo = (
        pd.read_parquet(RUTA_LISTA_CVEGEO)["CVEGEO"]
        .astype(str)
        .tolist()
    )

    df_in = pd.read_parquet(RUTA_DATASET_INCOMPLETO)
    df_in["CVEGEO"] = df_in["CVEGEO"].astype(str)

    # === CORREGIDO: LEER LOS NOMBRES REALES EN TU ARCHIVO ===
    with h5py.File(RUTA_MATRIZ_SIM, "r") as f:
        matriz = f["matriz"][:]    # ok
        cvegeo = f["cvegeo"][:]    # ok

    # convertir bytes → string
    cvegeo = [c.decode("utf-8") for c in cvegeo]

    # construir DataFrame cuadrado
    df_matriz = pd.DataFrame(matriz, index=cvegeo, columns=cvegeo)

    matriz_sim = df_matriz.copy()

    df_out = completar_dataset(df_in, matriz_sim, lista_cvegeo, top_n=5)

    df_out.to_parquet(RUTA_SALIDA, index=False)
    print(f"[OK] Guardado en {RUTA_SALIDA}")
