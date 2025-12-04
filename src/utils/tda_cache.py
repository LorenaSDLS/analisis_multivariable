# src/utils/tda_cache.py
import h5py
import numpy as np
from joblib import Parallel, delayed
from src.utils.tda_utils import normalizar_serie, takens_embedding, calcular_diagrama
import os

def generar_y_cachear_diagramas(df_series, columna, salida_h5, dim=3, delay=5, n_jobs=-1):
    """
    df_series: DataFrame con columnas ['cvegeo', 'fecha', columna]
    salida_h5: path al archivo h5 donde guardaremos una lista de diagramas (pickled bytes)
    """
    municipios = sorted(df_series["cvegeo"].unique())
    n = len(municipios)
    os.makedirs(os.path.dirname(salida_h5), exist_ok=True)

    def proc(cve):
        x = df_series[df_series.cvegeo == cve][columna].values
        x_norm = normalizar_serie(x)
        emb = takens_embedding(x_norm, delay=delay, dimension=dim)
        dgm = calcular_diagrama(emb)
        # serializar con numpy np.savez_compressed o pickle
        return cve, dgm

    results = Parallel(n_jobs=n_jobs)(
        delayed(proc)(cve) for cve in municipios
    )

    # Guardar como lista de pickles dentro de h5 (cada diagrama en bytes)
    import pickle
    with h5py.File(salida_h5, "w") as f:
        dt = h5py.special_dtype(vlen=bytes)
        grp = f.create_group("diagramas")
        for cve, dgm in results:
            grp.create_dataset(cve, data=pickle.dumps(dgm), dtype=dt)
        f.create_dataset("cvegeo", data=np.array(municipios, dtype='S5'))

    return salida_h5
