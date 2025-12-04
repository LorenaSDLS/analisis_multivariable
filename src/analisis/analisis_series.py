"""
ANÁLISIS DE SERIES DE TIEMPO
Define funciones de limpieza, validación y comparación mediante distancia L2.
"""

import numpy as np
import pandas as pd

# ==============================
#   VALIDACIONES Y PREPROCESO
# ==============================

def normalizar_cvegeo(cve):
    """Convierte cvegeo a string y rellena con ceros si es necesario."""
    return str(cve).strip().zfill(5)


def limpiar_serie(serie):
    """
    Convierte a valores numéricos y elimina NaN.
    Devuelve un vector numpy.
    """
    serie = pd.to_numeric(serie, errors="coerce")
    serie = serie.dropna()

    if len(serie) == 0:
        raise ValueError("La serie quedó vacía después de la limpieza.")

    return serie.to_numpy(dtype=float)


def alinear_series(s1, s2):
    """
    Asegura que ambas series tengan la misma longitud.
    Si no, recorta la más larga.
    """
    n = min(len(s1), len(s2))
    return s1[:n], s2[:n]


# ======================================
#   DISTANCIA L2 (EUCLIDIANA)
# ======================================

def distancia_l2(s1, s2):
    """Calcula la distancia L2 entre dos series numpy alineadas."""
    diff = s1 - s2
    return np.sqrt(np.sum(diff ** 2))


# ======================================
#   CONVERSIÓN A SIMILITUD 0–1
# ======================================

def distancia_a_similitud(distancia):
    """Convierte una distancia L2 en una similitud entre 0 y 1."""
    return 1 / (1 + distancia)


# ======================================
#   COMPARADOR PRINCIPAL
# ======================================

def comparar_series_arrays(arrA, arrB):
    """
    Compara directamente dos arrays numpy de series.
    """

    # Alinear
    n = min(len(arrA), len(arrB))
    s1 = arrA[:n]
    s2 = arrB[:n]

    # Distancia
    diff = s1 - s2
    d = np.sqrt(np.sum(diff ** 2))

    # Similitud
    sim = 1 / (1 + d)
    return sim
