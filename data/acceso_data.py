import pandas as pd
from src.utils.helpers import normalizar_cvegeo



def cargar_categoricos():
    topo = pd.read_csv('data/categoricos/mun_sist_topoformas.csv')
    topo["CVEGEO"] = topo["CVEGEO"].apply(normalizar_cvegeo)
    eda = pd.read_csv('data/categoricos/mun_edafologia.csv')
    eda["CVEGEO"] = eda["CVEGEO"].apply(normalizar_cvegeo)
    return eda, topo

def cargar_numericos():
    unidades_clima = pd.read_csv('data/numericos/mun_unidades_climaticas_final.csv')
    unidades_clima["CVEGEO"] = unidades_clima["CVEGEO"].apply(normalizar_cvegeo)
    temperatura_anual = pd.read_csv('data/numericos/mun_temp_media_anual.csv')
    temperatura_anual["CVEGEO"] = temperatura_anual["CVEGEO"].apply(normalizar_cvegeo)
    precipitacion_anual = pd.read_csv('data/numericos/mun_precip_media_anual.csv')
    precipitacion_anual["CVEGEO"] = precipitacion_anual["CVEGEO"].apply(normalizar_cvegeo)
    return precipitacion_anual, temperatura_anual, unidades_clima

def cargar_mensuales():
    radiacion = pd.read_parquet('data/series_tiempo/Radiacion_municipal.parquet')
    radiacion["CVEGEO"] = radiacion["CVEGEO"].apply(normalizar_cvegeo)
    sequia = pd.read_parquet('data/series_tiempo/Sequia_mensual_completa.parquet')
    sequia["CVEGEO"] = sequia["CVEGEO"].apply(normalizar_cvegeo)
    return radiacion, sequia

def cargar_municipios():
    municipios = pd.read_parquet('data/tabla_municipios.parquet')
    municipios["CVEGEO"] = municipios["CVEGEO"].apply(normalizar_cvegeo)
    return municipios

