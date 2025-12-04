import numpy as np
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import Scaler, PairwiseDistance

from sklearn.preprocessing import MinMaxScaler


# ======================================================
# EMBEDDING DE TAKENS MANUAL
# ======================================================
def takens_embedding(X, delay=1, dimension=3):
    N = len(X)
    M = N - (dimension - 1) * delay
    if M <= 0:
        raise ValueError("Serie demasiado corta para Takens embedding")

    emb = np.zeros((M, dimension), dtype=np.float32)
    for i in range(dimension):
        emb[:, i] = X[i * delay : i * delay + M]
    return emb


# ======================================================
# PERSISTENCIA (VR COMPLETAMENTE DISPONIBLE)
# ======================================================
vr = VietorisRipsPersistence(
    homology_dimensions=[0, 1],
    metric="euclidean",
    n_jobs=1
)

def calcular_diagrama(emb):
    return vr.fit_transform([emb])[0]

# ======================================================
# INTERFAZ COMPATIBLE PARA TDA MATRIZ
# ======================================================
def obtener_diagrama(emb):
    """
    Devuelve el diagrama de persistencia de un embedding ya construido.
    Simplemente envuelve a calcular_diagrama para mantener compatibilidad
    con el módulo tda_matriz.py
    """
    return calcular_diagrama(emb)


# ======================================================
# DISTANCIA ENTRE DIAGRAMAS (SIEMPRE DISPONIBLE)
# ======================================================
# PairwiseDistance por defecto usa "sliced_wasserstein"
dist_tda = PairwiseDistance(metric="sliced_wasserstein")


def distancia_diagramas(dgm1, dgm2):
    return float(dist_tda.fit_transform([dgm1, dgm2])[0, 1])


# ======================================================
# NORMALIZACIÓN
# ======================================================
def normalizar_serie(x):
    return MinMaxScaler().fit_transform(x.reshape(-1, 1)).flatten()


# ======================================================
# FUNCIÓN PRINCIPAL DE DISTANCIA TDA ENTRE SERIES
# ======================================================
def distancia_tda(serie1, serie2, delay=1, dimension=3):
    """
    Calcula distancia entre dos series usando TDA:
    - Normalización
    - Takens embedding
    - Vietoris-Rips
    - Sliced Wasserstein (siempre disponible)
    """
    s1 = normalizar_serie(serie1)
    s2 = normalizar_serie(serie2)

    emb1 = takens_embedding(s1, delay=delay, dimension=dimension)
    emb2 = takens_embedding(s2, delay=delay, dimension=dimension)

    dgm1 = calcular_diagrama(emb1)
    dgm2 = calcular_diagrama(emb2)

    return distancia_diagramas(dgm1, dgm2)
