import time
import numpy as np
import psutil
import h5py
from src.utils.helpers import Progreso
from multiprocessing import cpu_count
from src.construccion_matriz.matriz_mensual import construir_matriz_optimizada


def benchmark_matriz_mensual(nombre, parquet, salida_h5, normalizar=False):
    print(f"\n=== Benchmark: {nombre.upper()} ===")

    inicio = time.perf_counter()
    construir_matriz_optimizada(
        path_parquet=parquet,
        salida_h5=salida_h5,
        normalizar=normalizar
    )
    fin = time.perf_counter()
    tiempo_total = fin - inicio

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
        nombre="precipitacion_diaria",
        parquet="data/prec_diario.parquet",
        salida_h5="outputs/matriz_precipitacion.h5",
        normalizar=True
    )

    benchmark_matriz_mensual(
        nombre="temperatura_diaria",
        parquet="data/temp_diario.parquet",
        salida_h5="outputs/matriz_temperatura.h5",
        normalizar=False
    )

    print("\n===== BENCHMARK TERMINADO =====\n")


if __name__ == "__main__":
    main()
