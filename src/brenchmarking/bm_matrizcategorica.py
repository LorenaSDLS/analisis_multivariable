#python3 -m src.brenchmarking.bm_matrizcategorica
import time
import h5py
import numpy as np
from data.acceso_data import cargar_categoricos
from src.construccion_matriz.matriz_categorica import construir_matriz_categorica
from src.utils.helpers import Progreso
from src.utils.rendimiento import * 
import psutil
from src.utils.helpers import Progreso
uso_cpu = psutil.cpu_percent(interval=1, percpu=False)


def main():
    # === CARGAR DATOS ===
    eda, topo = cargar_categoricos()
    lista_cvegeo = sorted(set(eda["CVEGEO"]) & set(topo["CVEGEO"]))
    num_municipios = len(lista_cvegeo)
    print(f"Municipios cargados: {num_municipios}")

    # === CREAR BARRA DE PROGRESO ===
    progreso = Progreso(total=num_municipios, cada=50, etiqueta="Matriz cat.")

    # === MEDIR TIEMPO DE EJECUCIÓN ===
    inicio = time.perf_counter()
    matriz = construir_matriz_categorica(
        lista_cvegeo,
        edafologia=eda,
        topoforma=topo,
        n_jobs=-1,       # Paralelización
        cada=50,
        etiqueta="Matriz cat."
    )
    fin = time.perf_counter()
    tiempo_total = fin - inicio

    # === GUARDAR MATRIZ ===
    # === GUARDAR MATRIZ ===
    with h5py.File("outputs/matriz_categorica.h5", "w") as f:
        f.create_dataset("matriz", data=matriz)
        # Convertir CVEGEO a bytes para HDF5
        lista_bytes = np.array(lista_cvegeo, dtype="S5")
        f.create_dataset("cvegeo", data=lista_bytes)


    # === MEDIR MEMORIA Y CPU FINAL ===
    proceso = psutil.Process()
    memoria_actual = proceso.memory_info().rss / 1e6  # MB
    cpu_percent = proceso.cpu_percent(interval=0.1)  # instante, solo como referencia
    uso_cpu = psutil.cpu_percent(interval=1, percpu=False)

    # === REPORTAR ===
    print(f"\nMatriz construida en {tiempo_total:.2f} segundos.")
    print(f"Memoria usada (aprox.): {memoria_actual:.2f} MB")
    print(f"Uso CPU (1 seg): {uso_cpu:.1f}%")

if __name__ == "__main__":
    main()
