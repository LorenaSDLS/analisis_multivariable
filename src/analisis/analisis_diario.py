import pandas as pd
import numpy as np

# ============================================================
# Detecta si una columna de fecha contiene hora
# ============================================================
def tiene_hora(serie_fecha: pd.Series) -> bool:
    """
    Determina si una columna de fechas contiene información de horas.
    """
    return serie_fecha.dt.hour.nunique() > 1 or serie_fecha.dt.minute.nunique() > 1


# ============================================================
# Convierte datos horarios a datos diarios (sumando o promediando)
# ============================================================
def convertir_a_diario(df, fecha_col, id_col, var_col, metodo="sum"):
    """
    Convierte datos con fecha-hora a datos diarios.
    metodo: 'sum' (precipitación, evaporación) o 'mean' (temperaturas)
    """
    df["fecha"] = df[fecha_col].dt.date

    if metodo == "sum":
        df_diario = df.groupby([id_col, "fecha"])[var_col].sum().reset_index()
    else:
        df_diario = df.groupby([id_col, "fecha"])[var_col].mean().reset_index()

    # Renombra fecha → mantener consistencia
    df_diario.rename(columns={"fecha": fecha_col}, inplace=True)
    df_diario[fecha_col] = pd.to_datetime(df_diario[fecha_col])

    return df_diario


# ============================================================
# Interpolación horizontal por municipio
# ============================================================
def interpolar_por_municipio(df, id_col, fecha_col, var_col):
    """
    Interpola valores faltantes por municipio.
    Supone que df está ordenado por fecha.
    """
    return (
        df.set_index(fecha_col)
          .groupby(id_col)[var_col]
          .apply(lambda s: s.interpolate(method="linear"))
          .reset_index()
    )


# ============================================================
# ANÁLISIS PRINCIPAL
# ============================================================
def analizar_series_diarias(
    path_parquet: str,
    var_col: str,
    id_col: str = "CVEGEO",
    fecha_col: str = "valid_time",
    output_path: str = "series_limpias.parquet",
    metodo_horario="auto"
):
    """
    1. Carga un parquet con series diarias.
    2. Detecta si las fechas incluyen hora; si sí → convierte a diario.
    3. Ordena cronológicamente.
    4. Revisa valores faltantes.
    5. Interpola horizontalmente por municipio.
    6. Guarda parquet limpio.
    """

    print("-----------------------------------------------------------")
    print(f"Cargando archivo: {path_parquet}")
    df = pd.read_parquet(path_parquet)

    # Asegurar formato datetime
    df[fecha_col] = pd.to_datetime(df[fecha_col], errors="coerce")

    # Orden básico antes de cualquier cosa
    df = df.sort_values(by=[id_col, fecha_col])

    # =======================================================
    # 1. Detectar si tiene hora
    # =======================================================
    print("Detectando si la columna de fechas contiene hora...")

    if tiene_hora(df[fecha_col]):
        print("→ Tiene hora. Convirtiendo a formato diario...")

        # Selección automática de método según variable
        if metodo_horario == "auto":
            if "prec" in var_col.lower():
                metodo = "sum"
            elif "evap" in var_col.lower() or "eva" in var_col.lower():
                metodo = "sum"
            else:
                metodo = "mean"
        else:
            metodo = metodo_horario

        df = convertir_a_diario(df, fecha_col, id_col, var_col, metodo)
        print(f"Conversión completada con método: {metodo}")

    else:
        print("→ Ya es diario. No se requiere conversión.")

    # =======================================================
    # 2. Revisar valores faltantes
    # =======================================================
    print("Revisando valores faltantes...")
    faltantes = df[var_col].isna().sum()
    print(f"Valores faltantes antes de interpolar: {faltantes}")

    # =======================================================
    # 3. Interpolación horizontal por municipio
    # =======================================================
    print("Realizando interpolación por municipio...")
    df_interp = interpolar_por_municipio(df, id_col, fecha_col, var_col)

    faltantes_final = df_interp[var_col].isna().sum()
    print(f"Valores faltantes después de interpolación: {faltantes_final}")

    # =======================================================
    # 4. Guardar archivo limpio
    # =======================================================
    print(f"Guardando archivo limpio en: {output_path}")
    df_interp.to_parquet(output_path)
    print("Proceso completado.")
    print("-----------------------------------------------------------")

    return df_interp


# ============================================================
# Script ejecutable
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Análisis de series diarias con conversión si es necesario.")
    parser.add_argument("--input", required=True, help="Ruta del parquet de entrada")
    parser.add_argument("--variable", required=True, help="Nombre de la columna de variable (prec, t2m, evavt, etc.)")
    parser.add_argument("--output", required=True, help="Ruta del parquet limpio de salida")

    args = parser.parse_args()

    analizar_series_diarias(
        path_parquet=args.input,
        var_col=args.variable,
        output_path=args.output
    )
