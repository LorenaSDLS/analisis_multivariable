import time
import numpy as np
import psutil
import h5py
from src.utils.helpers import Progreso
from multiprocessing import cpu_count
from src.construccion_matriz.matriz_mensual import construir_matriz_optimizada

# ============================================================
# BENCHMARK PARA MATRICES MENSUALES (RADIACIÓN / SEQUÍA / ETC)
# ============================================================

def benchmark_matriz_mensual(nombre, parquet, salida_h5, normalizar=False):
    print(f"\n=== Benchmark: {nombre.upper()} ===")

    # ====== MEDIR TIEMPO ======
    inicio = time.perf_counter()

    # Progreso (municipios se sabrá dentro de la función)
    print("Iniciando construcción optimizada...")

    construir_matriz_optimizada(
        path_parquet=parquet,
        salida_h5=salida_h5,
        normalizar=normalizar
    )

    fin = time.perf_counter()
    tiempo_total = fin - inicio

    # ====== METRICAS DE SISTEMA ======
    proceso = psutil.Process()
    memoria = proceso.memory_info().rss / 1024**2  # MB
    cpu = psutil.cpu_percent(interval=1)

    print(f"\n💠 RESULTADOS {nombre.upper()} 💠")
    print(f"Tiempo total     : {tiempo_total:.2f} s")
    print(f"Uso de memoria   : {memoria:.2f} MB")
    print(f"Uso de CPU (1 s) : {cpu:.1f}%")
    print(f"Matriz guardada en: {salida_h5}")
    print("===========================================\n")


def main():
    print("\n===== BENCHMARKING MATRICES MENSUALES =====\n")

    benchmark_matriz_mensual(
        nombre="radiación",
        parquet="data/series_tiempo/Radiacion_municipal_limpia.parquet",
        salida_h5="outputs/matriz_radiacion.h5",
        normalizar=True
    )

    benchmark_matriz_mensual(
        nombre="sequía",
        parquet="data/series_tiempo/Sequia_municipal_limpia.parquet",
        salida_h5="outputs/matriz_sequia.h5",
        normalizar=False
    )

    print("\n===== BENCHMARK TERMINADO =====\n")


if __name__ == "__main__":
    main()
