import pandas as pd
from pathlib import Path

def limpiar_series_mensuales_sequia(path_entrada, path_salida):
    df = pd.read_parquet(path_entrada)


    # 1. Detectar columnas que son meses con formato YYYY-MM
    columnas_mes = [
        c for c in df.columns
        if isinstance(c, str) and len(c) == 7 and c[:4].isdigit() and c[4] == "-" and c[5:7].isdigit()
    ]

    # Ordenar las columnas de meses cronológicamente
    columnas_mes = sorted(columnas_mes)

    # Reporte inicial
    reporte = {
        "input_rows": df.shape[0],
        "input_cols": df.shape[1],
        "meses_detectados": len(columnas_mes),
        "meses_lista_sample": columnas_mes[:5],
        "num_municipios": df.shape[0],
        "num_meses": len(columnas_mes),
    }

    # 2. Verificar faltantes
    reporte["municipios_con_faltantes"] = df[columnas_mes].isna().any(axis=1).sum()
    reporte["meses_con_faltantes"] = df[columnas_mes].isna().any(axis=0).sum()

    # (Opcional) Rellenar faltantes con interpolación por municipio
    df[columnas_mes] = df[columnas_mes].interpolate(axis=1, limit_direction="both")

    # 3. Guardar en parquet
    Path(path_salida).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path_salida, index=False)

    reporte["guardado_en"] = path_salida
    return reporte


