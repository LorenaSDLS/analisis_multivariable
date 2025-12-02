import numpy as np
from src.utils.helpers import normalizar_cvegeo, Progreso
from joblib import Parallel, delayed
from src.analisis.analisis_categorico import comparar_edafologia, comparar_topoforma, unir_edafologia, limpiar_texto


def construir_matriz_categorica(lista_cvegeo, edafologia, topoforma, n_jobs=-1, cada=50, etiqueta="Matriz cat."):
    n = len(lista_cvegeo)
    matriz = np.zeros((n, n))

    # --- PRE-INDEXAR LOS DATOS ---
    eda_dict = {cve: unir_edafologia(row) 
                for cve, row in edafologia.set_index("CVEGEO").iterrows()}
    topo_dict = {cve: limpiar_texto(row["CLAVE"]) 
                 for cve, row in topoforma.set_index("CVEGEO").iterrows()}

    # --- FUNCION DE COMPARACION PARA UN PAR ---
    def sim_pair(i, j):
        mun_i = lista_cvegeo[i]
        mun_j = lista_cvegeo[j]

        val1_eda = eda_dict.get(mun_i, "")
        val2_eda = eda_dict.get(mun_j, "")
        val1_topo = topo_dict.get(mun_i, "")
        val2_topo = topo_dict.get(mun_j, "")

        sim_eda = comparar_edafologia(val1_eda, val2_eda)
        sim_topo = comparar_topoforma(val1_topo, val2_topo)
        sim = (sim_eda + sim_topo) / 2
        return i, j, sim

    # --- PREPARAR BARRA DE PROGRESO ---
    total_pairs = n * (n + 1) // 2
    progreso = Progreso(total=total_pairs, cada=cada, etiqueta=etiqueta)
    count = 0

    # --- CALCULAR MATRIZ PARALLEL ---
    def callback(result):
        nonlocal count
        i, j, sim = result
        matriz[i, j] = sim
        matriz[j, i] = sim
        count += 1
        if count % cada == 0:
            progreso.paso(count)

    # --- EJECUTAR JOBS ---
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(sim_pair)(i, j) for i in range(n) for j in range(i, n)
    )

    # --- RECONSTRUIR MATRIZ Y BARRA PROGRESO ---
    for res in results:
        callback(res)

    return matriz

