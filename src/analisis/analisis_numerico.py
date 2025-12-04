import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from data.acceso_data import cargar_numericos, cargar_municipios
from src.utils.helpers import normalizar_cvegeo, Progreso

# =======================================================
# MODELOS DE SIMILITUD NUMÉRICA
# =======================================================

def similitud_proporcional(x1, x2):
    """
    Calcula similitud proporcional entre dos valores numéricos.
    """
    if x1 == 0 and x2 == 0:
        return 1.0
    max_val = max(abs(x1), abs(x2))
    diff = abs(x1 - x2)
    return 1 - (diff / max_val) if max_val != 0 else 0.0

def es_numerico(valor):
    try:
        float(valor)
        return True
    except:
        return False

def es_rango(valor):
    return isinstance(valor, str) and "-" in valor

def rango_a_promedio(rango):
    try:
        min_val, max_val = [float(v.strip()) for v in rango.split("-")]
        return (min_val + max_val) / 2
    except:
        return 0.0

def comparar_valores_num(v1, v2):
    """
    Compara dos valores numéricos o rangos y devuelve similitud [0,1].
    """
    if es_numerico(v1) and es_numerico(v2):
        return similitud_proporcional(float(v1), float(v2))
    if es_rango(v1) and es_rango(v2):
        return similitud_proporcional(rango_a_promedio(v1), rango_a_promedio(v2))
    if es_rango(v1) and es_numerico(v2):
        return similitud_proporcional(rango_a_promedio(v1), float(v2))
    if es_rango(v2) and es_numerico(v1):
        return similitud_proporcional(float(v1), rango_a_promedio(v2))
    return 0.0

# =======================================================
# CARGA DE DATOS
# =======================================================
precipitacion, temperatura, unidades_climaticas = cargar_numericos()
municipios = cargar_municipios()

# =======================================================
# COMPARADOR PRINCIPAL
# =======================================================
def comparar_municipios_num(mun1, mun2):
    """
    Compara dos municipios usando solo datos numéricos:
    precipitación, temperatura y unidad climática.
    """
    mun1 = normalizar_cvegeo(mun1)
    mun2 = normalizar_cvegeo(mun2)

    fila_prec1 = precipitacion[precipitacion["CVEGEO"] == mun1]
    fila_prec2 = precipitacion[precipitacion["CVEGEO"] == mun2]

    fila_temp1 = temperatura[temperatura["CVEGEO"] == mun1]
    fila_temp2 = temperatura[temperatura["CVEGEO"] == mun2]

    fila_uni1 = unidades_climaticas[unidades_climaticas["CVEGEO"] == mun1]
    fila_uni2 = unidades_climaticas[unidades_climaticas["CVEGEO"] == mun2]

    if fila_prec1.empty or fila_prec2.empty:
        return None
    if fila_temp1.empty or fila_temp2.empty:
        return None
    if fila_uni1.empty or fila_uni2.empty:
        return None

    v_prec1 = fila_prec1.iloc[0]["RANGOS"]
    v_prec2 = fila_prec2.iloc[0]["RANGOS"]

    v_temp1 = fila_temp1.iloc[0]["RANGOS"]
    v_temp2 = fila_temp2.iloc[0]["RANGOS"]

    v_uni1 = fila_uni1.iloc[0]["TIPO_N"]
    v_uni2 = fila_uni2.iloc[0]["TIPO_N"]

    sim_prec = comparar_valores_num(v_prec1, v_prec2)
    sim_temp = comparar_valores_num(v_temp1, v_temp2)
    sim_uni  = similitud_proporcional(v_uni1, v_uni2)

    return (sim_prec + sim_temp + sim_uni) / 3 # Promedio de similitudes

 