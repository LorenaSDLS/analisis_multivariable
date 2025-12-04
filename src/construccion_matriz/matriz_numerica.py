import numpy as np
from joblib import Parallel, delayed
from src.utils.helpers import normalizar_cvegeo, Progreso
from src.analisis.analisis_numerico import similitud_proporcional, rango_a_promedio, similitud_proporcional

def construir_matriz_numerica(lista_cvegeo, precipitacion, temperatura, unidades_climaticas, n_jobs=-1, cada=50, etiqueta="Matriz num."):
    """
    Construye la matriz de similitud numérica entre municipios.
    """
    n = len(lista_cvegeo)
    matriz = np.zeros((n, n))

    # --- PRE-INDEXAR Y PREPROCESAR LOS DATOS ---
    prec_dict = {cve: rango_a_promedio(fila["RANGOS"]) 
                 for cve, fila in precipitacion.set_index("CVEGEO").iterrows()}
    temp_dict = {cve: rango_a_promedio(fila["RANGOS"]) 
                 for cve, fila in temperatura.set_index("CVEGEO").iterrows()}
    uni_dict = {cve: fila["TIPO_N"] for cve, fila in unidades_climaticas.set_index("CVEGEO").iterrows()}

    # --- FUNCION DE COMPARACION PARA UN PAR ---
    def sim_pair(i, j):
        mun_i = lista_cvegeo[i]
        mun_j = lista_cvegeo[j]

        sim_prec = similitud_proporcional(prec_dict.get(mun_i, 0.0), prec_dict.get(mun_j, 0.0))
        sim_temp = similitud_proporcional(temp_dict.get(mun_i, 0.0), temp_dict.get(mun_j, 0.0))
        sim_uni  = similitud_proporcional(uni_dict.get(mun_i, 0.0), uni_dict.get(mun_j, 0.0))

        sim_final = (sim_prec + sim_temp + sim_uni) / 3
        return i, j, sim_final

    # --- BARRA DE PROGRESO ---
    total_pairs = n * (n + 1) // 2
    progreso = Progreso(total=total_pairs, cada=cada, etiqueta=etiqueta)
    count = 0

    # --- CALLBACK PARA ACTUALIZAR MATRIZ Y BARRA ---
    def callback(result):
        nonlocal count
        i, j, sim = result
        matriz[i, j] = sim
        matriz[j, i] = sim
        count += 1
        if count % cada == 0:
            progreso.paso(count)

    # --- PARALLEL COMPUTATION ---
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(sim_pair)(i, j) for i in range(n) for j in range(i, n)
    )

    for res in results:
        callback(res)

    return matriz

