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

def cargar_series_tiempo():
    series_tiempo = pd.read_parquet('data/series_tiempo/mun_series_tiempo.parquet')
    series_tiempo["CVEGEO"] = series_tiempo["CVEGEO"].apply(normalizar_cvegeo)
    return series_tiempo

def cargar_municipios():
    municipios = pd.read_parquet('data/tabla_municipios.parquet')
    municipios["CVEGEO"] = municipios["CVEGEO"].apply(normalizar_cvegeo)
    return municipios