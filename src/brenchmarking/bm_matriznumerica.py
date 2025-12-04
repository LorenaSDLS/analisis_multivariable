# python3 -m src.brenchmarking.bm_matriznumerica
import time
import numpy as np
import psutil
import h5py
from src.utils.helpers import Progreso
from data.acceso_data import cargar_numericos, cargar_municipios
from src.construccion_matriz.matriz_numerica import construir_matriz_numerica

def main():
    # === CARGAR DATOS ===
    precipitacion, temperatura, unidades_climaticas = cargar_numericos()
    lista_cvegeo = sorted(set(precipitacion["CVEGEO"]) &
                          set(temperatura["CVEGEO"]) &
                          set(unidades_climaticas["CVEGEO"]))
    
    num_municipios = len(lista_cvegeo)
    print(f"Municipios cargados: {num_municipios}")

    # === CREAR BARRA DE PROGRESO ===
    progreso = Progreso(total=num_municipios, cada=50, etiqueta="Matriz num.")

    # === MEDIR TIEMPO DE EJECUCIÓN ===
    inicio = time.perf_counter()
    matriz = construir_matriz_numerica(
        lista_cvegeo,
        precipitacion=precipitacion,
        temperatura=temperatura,
        unidades_climaticas=unidades_climaticas,
        n_jobs=-1,
        cada=50,
        etiqueta="Matriz num."
    )
    fin = time.perf_counter()
    tiempo_total = fin - inicio

    # === GUARDAR MATRIZ ===
    with h5py.File("outputs/matriz_numerica.h5", "w") as f:
        f.create_dataset("matriz", data=matriz)
        # Convertir CVEGEO a bytes para HDF5
        lista_bytes = np.array(lista_cvegeo, dtype="S5")
        f.create_dataset("cvegeo", data=lista_bytes)

    # === MEDIR MEMORIA Y CPU FINAL ===
    proceso = psutil.Process()
    memoria_actual = proceso.memory_info().rss / 1e6  # MB
    uso_cpu = psutil.cpu_percent(interval=1, percpu=False)

    # === REPORTAR ===
    print(f"\nMatriz construida en {tiempo_total:.2f} segundos.")
    print(f"Memoria usada (aprox.): {memoria_actual:.2f} MB")
    print(f"Uso CPU (1 seg): {uso_cpu:.1f}%")

if __name__ == "__main__":
    main()
