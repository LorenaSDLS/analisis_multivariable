# src/bench/bench_tda.py
import time, psutil
from src.utils.tda_cache import generar_y_cachear_diagramas
from src.construccion_matriz.tda_matriz import construir_matriz_tda_from_cache
from src.utils.tda_vector import diagramas_a_images
from sklearn.metrics import pairwise_distances
import h5py

def bench(df_series, columna):
    t0 = time.perf_counter()
    cache = "data/tda/cache_diagramas.h5"
    generar_y_cachear_diagramas(df_series, columna, cache, n_jobs=-1)
    t1 = time.perf_counter()
    print("Tiempo calcular diagramas:", t1 - t0)

    # cargar diagramas (ejemplo)
    import pickle
    with h5py.File(cache, "r") as f:
        cves = [c.decode() if isinstance(c, bytes) else c for c in f["cvegeo"][:]]
        grp = f["diagramas"]
        diagramas = [pickle.loads(grp[c][()]) for c in cves]

    t2 = time.perf_counter()
    print("Tiempo cargar diagramas:", t2 - t1)

    X_images = diagramas_a_images(diagramas, n_jobs=-1)
    t3 = time.perf_counter()
    print("Tiempo persistence images:", t3 - t2)

    D = pairwise_distances(X_images, metric='euclidean', n_jobs=-1).astype('float32')
    t4 = time.perf_counter()
    print("Tiempo pairwise distances:", t4 - t3)

    proceso = psutil.Process()
    print("Memoria final (MB):", proceso.memory_info().rss / 1e6)
