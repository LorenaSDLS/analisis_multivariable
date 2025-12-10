import h5py
import pandas as pd
import numpy as np
import networkx as nx

def cargar_h5(path):
    with h5py.File(path, "r") as f:
        matriz = f["matriz"][:]
        cvegeo = [c.decode("utf-8") for c in f["cvegeo"][:]]
    return pd.DataFrame(matriz, index=cvegeo, columns=cvegeo)

df_sim = cargar_h5("outputs/matriz_sim_general.h5")




def construir_grafo_knn(df_sim, k=10):
    """
    Convierte una matriz de similitud en un grafo k-NN.
    - df_sim: DataFrame cuadrado de similitud.
    - k: número de vecinos más cercanos.
    
    Las aristas tienen peso = similitud (entre 0 y 1).
    """

    G = nx.Graph()
    nodos = df_sim.index.tolist()

    # Agregar nodos
    for n in nodos:
        G.add_node(n)

    # Para cada municipio, agregar sus k vecinos con más similitud
    for i, fila in df_sim.iterrows():
        
        # ordena de mayor a menor similitud (excluyendo a sí mismo)
        vecinos = fila.drop(i).nlargest(k)

        for j, sim in vecinos.items():
            # las aristas son no dirigidas, pero evita duplicarlas
            if not G.has_edge(i, j):
                G.add_edge(i, j, weight=float(sim))

    return G

G = construir_grafo_knn(df_sim, k=10)
nx.write_graphml(G, "outputs/grafo_knn.graphml")
