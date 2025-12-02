import Levenshtein
import pandas as pd
import numpy as np
from data.acceso_data import cargar_categoricos, cargar_municipios

# =======================================================

# NORMALIZADORES
# =======================================================

def normalizar_cvegeo(valor):
    """Convierte un CVEGEO a string de 5 dígitos."""
    try:
        return str(valor).zfill(5)
    except:
        return None


def limpiar_texto(valor):
    """Convierte cualquier valor a string seguro y limpio."""
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return ""
    return str(valor).strip().lower()


# =======================================================
# MODELOS DE SIMILITUD
# =======================================================

def comparar_edafologia(text1, text2):
    """
    Compara dos textos de edafología con Levenshtein.
    Devuelve similitud entre 0 y 1.
    """
    t1 = limpiar_texto(text1)
    t2 = limpiar_texto(text2)

    if not t1 or not t2:
        return 0.0

    return Levenshtein.ratio(t1, t2)


def comparar_topoforma(text1, text2):
    """
    Compara claves topográficas con Jaro-Winkler.
    Devuelve similitud entre 0 y 1.
    """
    t1 = limpiar_texto(text1)
    t2 = limpiar_texto(text2)

    if not t1 or not t2:
        return 0.0

    return Levenshtein.jaro_winkler(t1, t2)


# =======================================================
# UNIR CAMPOS DE EDAFOLOGÍA
# =======================================================

def unir_edafologia(fila):
    """Concatena los campos relevantes en un solo texto grande."""
    columnas = ["CLAVE_WRB", "GRUPO1", "GRUPO2", "GRUPO3", "CLASE_TEXT", "FRUDICA"]
    return " ".join([limpiar_texto(fila[col]) for col in columnas])


# =======================================================
# CARGA DE DATOS
# =======================================================
    
edafologia, topoforma = cargar_categoricos()
municipios = cargar_municipios()

# =======================================================
# MODELO GENERAL
# =======================================================

def modelo_gral(val1_eda, val2_eda, val1_topo, val2_topo):
    sim_eda = comparar_edafologia(val1_eda, val2_eda)
    sim_topo = comparar_topoforma(val1_topo, val2_topo)
    return (sim_eda + sim_topo) / 2


# =======================================================
# COMPARADOR PRINCIPAL
# =======================================================

def comparar_municipios(mun1, mun2):
    """
    Compara dos municipios considerando:
    - Edafología
    - Topoforma
    Retorna una similitud [0,1].
    """

    mun1 = normalizar_cvegeo(mun1)
    mun2 = normalizar_cvegeo(mun2)

    fila_eda_m1 = edafologia[edafologia["CVEGEO"] == mun1]
    fila_eda_m2 = edafologia[edafologia["CVEGEO"] == mun2]
    fila_topo_m1 = topoforma[topoforma["CVEGEO"] == mun1]
    fila_topo_m2 = topoforma[topoforma["CVEGEO"] == mun2]

    if fila_eda_m1.empty or fila_eda_m2.empty:
        return None
    if fila_topo_m1.empty or fila_topo_m2.empty:
        return None

    v1_eda = unir_edafologia(fila_eda_m1.iloc[0])
    v2_eda = unir_edafologia(fila_eda_m2.iloc[0])

    v1_topo = limpiar_texto(fila_topo_m1.iloc[0]["CLAVE"])
    v2_topo = limpiar_texto(fila_topo_m2.iloc[0]["CLAVE"])

    return modelo_gral(v1_eda, v2_eda, v1_topo, v2_topo)


# =======================================================
# COMPARACIÓN DETALLADA
# =======================================================

def comparar_municipios_detallado(mun1, mun2):
    mun1 = normalizar_cvegeo(mun1)
    mun2 = normalizar_cvegeo(mun2)

    fila_eda_m1 = edafologia[edafologia["CVEGEO"] == mun1]
    fila_eda_m2 = edafologia[edafologia["CVEGEO"] == mun2]
    fila_topo_m1 = topoforma[topoforma["CVEGEO"] == mun1]
    fila_topo_m2 = topoforma[topoforma["CVEGEO"] == mun2]

    if fila_eda_m1.empty or fila_eda_m2.empty:
        print("No hay datos de edafología.")
        return
    if fila_topo_m1.empty or fila_topo_m2.empty:
        print("No hay datos de topoforma.")
        return

    v1_eda = unir_edafologia(fila_eda_m1.iloc[0])
    v2_eda = unir_edafologia(fila_eda_m2.iloc[0])

    v1_topo = limpiar_texto(fila_topo_m1.iloc[0]["CLAVE"])
    v2_topo = limpiar_texto(fila_topo_m2.iloc[0]["CLAVE"])

    sim_eda = comparar_edafologia(v1_eda, v2_eda)
    sim_topo = comparar_topoforma(v1_topo, v2_topo)
    sim_final = (sim_eda + sim_topo) / 2

    return f"""
============================================
Comparación entre {mun1} y {mun2}
============================================

EDAFOLOGÍA:
  {v1_eda}
  {v2_eda}
  → {sim_eda:.3f}

TOPOFORMA:
  {v1_topo} vs {v2_topo}
  → {sim_topo:.3f}

--------------------------------------------
SIMILITUD FINAL PROMEDIO → {sim_final:.3f}
--------------------------------------------
"""
