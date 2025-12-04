# src/brenchmarking/bm_matrizseries_blocks.py
import time
import psutil
import pandas as pd
import h5py
import numpy as np

from src.construccion_matriz.matriz_blocks import construir_matriz_similitud_blocks

def main():
    print("\n=== BENCHMARKING MATRIZ POR BLOQUES (L2) ===\n")

    # Carga (archivo WIDE mensual en tu caso)
    print("Cargando series...")
    df = pd.read_parquet("data/series_tiempo/Sequia_mensual_completa.parquet")
    indices = df["CVEGEO"].astype(str).values

    # Eliminar columnas basura si las hay
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # Asegurar índice CVEGEO
    if "CVEGEO" in df.columns:
        df["CVEGEO"] = df["CVEGEO"].astype(str).str.zfill(5)
        df = df.set_index("CVEGEO")
    else:
        # si ya está indexado, aseguramos formato
        df.index = df.index.astype(str).str.zfill(5)

    # Asegurar orden de columnas (YYYY-MM lexicográfico funciona)
    df_wide = df.copy()
    df_wide = df_wide.sort_index(axis=1)

    # Relleno de faltantes horizontalmente (no eliminamos municipios)
    df_wide = df_wide.ffill(axis=1).bfill(axis=1)

    lista_mun = sorted(df_wide.index.tolist())
    print(f"Municipios detectados: {len(lista_mun)}")
    print(f"Fechas por municipio: {df_wide.shape[1]}")

    # Medición inicial
    proceso = psutil.Process()
    memoria_ini = proceso.memory_info().rss / 1e6

    # Parámetros
    block_size = 250    # ajusta: 200-500 según RAM
    n_procs = 6         # ajusta al número de cores (no pongas más que cores físicos)

    # Ejecutar
    t0 = time.time()
    lista, matriz = construir_matriz_similitud_blocks(
        df_wide,
        lista_mun,
        block_size=block_size,
        n_procs=n_procs,
        use_parallel=True
    )
    dur = time.time() - t0

    memoria_fin = proceso.memory_info().rss / 1e6
    cpu_porcentaje = proceso.cpu_percent(interval=0.1)

    # Guardar
    print("Guardando matriz en HDF5...")
    with h5py.File("matrizsimilitud_sequia.h5", "w") as f:
        f.create_dataset("similitud", data=matriz)

        # Guardamos el MISMO orden usado en la matriz
        f.create_dataset("index", data=np.array(lista, dtype="S10"))
        f.create_dataset("municipios", data=np.array(lista, dtype="S50"))
# Reporte
    print("\n=== REPORTE ===")
    print(f"Municipios: {len(lista)}")
    print(f"Block size: {block_size}, n_procs: {n_procs}")
    print(f"Tiempo total: {dur:.2f} s")
    print(f"Memoria inicial: {memoria_ini:.2f} MB")
    print(f"Memoria final: {memoria_fin:.2f} MB")
    print(f"Uso CPU (instant): {cpu_porcentaje:.2f}%")
    print("Archivo: matriz_sim_sequia.h5")

if __name__ == "__main__":
    main()
