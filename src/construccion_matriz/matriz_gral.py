import h5py
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# ============================================================
# Función para cargar matriz y CVEGEO desde un archivo .h5
# ============================================================

# ============================================================
# Construcción de matriz general de similitud
# ============================================================
def cargar_matriz_y_cvegeo(path):
    """
    Carga la matriz y cvegeo desde un archivo HDF5,
    detectando automáticamente el nombre correcto de los datasets.
    """
    import h5py
    import numpy as np

    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        print(f"[INFO] Leyendo {path} — datasets: {keys}")

        # -------- Buscar dataset de matriz --------
        if "matriz" in keys:
            matriz = f["matriz"][:]
        elif "similitud" in keys:
            matriz = f["similitud"][:]
        else:
            raise KeyError(
                f"ERROR: No se encontró dataset de matriz en {path}. "
                "Se esperaba uno de: ['matriz', 'similitud']"
            )

        # -------- Buscar dataset de CVEGEO --------
        if "cvegeo" in keys:
            cvegeo = f["cvegeo"][:]
        elif "municipios" in keys:
            cvegeo = f["municipios"][:]
        else:
            raise KeyError(
                f"ERROR: No se encontró dataset de CVEGEO en {path}. "
                "Se esperaba uno de: ['cvegeo', 'municipios']"
            )

    return matriz, cvegeo

def matriz_similitud_general(paths, pesos, salida_h5):

    if len(paths) != len(pesos):
        raise ValueError("Error: paths y pesos deben tener el mismo largo")

    # Normalizar pesos a suma = 1
    pesos = np.array(pesos, dtype=np.float32)
    pesos = pesos / pesos.sum()

    # ---------------------------------
    # Cargar primera matriz para iniciar
    # ---------------------------------
    matriz0, cvegeo = cargar_matriz_y_cvegeo(paths[0])
    n = matriz0.shape[0]

    print(f"→ Cargando matriz base: {paths[0]}  |  Tamaño: {n} x {n}")

    matriz_final = matriz0 * pesos[0]

    # ---------------------------------
    # Acumular cada matriz ponderada
    # ---------------------------------
    for path, w in zip(paths[1:], pesos[1:]):
        print(f"→ Sumando matriz: {path}  | peso = {w}")
        M, cve = cargar_matriz_y_cvegeo(path)

        if M.shape != matriz_final.shape:
            raise ValueError(f"Dimensiones incompatibles: {path}")

        matriz_final += M * w

        # Verificar que CVEGEO coincida
        if not np.array_equal(cvegeo, cve):
            print("⚠ Advertencia: Los CVEGEO no son idénticos entre matrices")

    print("→ Matrices sumadas correctamente.")

    # ---------------------------------
    # Normalizar la matriz 0–1
    # ---------------------------------
    print("→ Normalizando matriz final…")
    scaler = MinMaxScaler()
    matriz_flat = matriz_final.reshape(-1, 1)
    matriz_norm = scaler.fit_transform(matriz_flat).reshape(n, n)

    # ---------------------------------
    # Convertir distancia → similitud
    # ---------------------------------
    matriz_sim = 1 - matriz_norm

    # ---------------------------------
    # Guardar archivo final
    # ---------------------------------
    with h5py.File(salida_h5, "w") as f:
        f.create_dataset("matriz", data=matriz_sim.astype(np.float32), compression="gzip")
        f.create_dataset("cvegeo", data=np.array(cvegeo, dtype="S"), compression="gzip")

    print("\n✔ MATRIZ FINAL GUARDADA")
    print("Archivo:", salida_h5)
    print("Dimensiones:", matriz_sim.shape)
    print("Similitud min/max:", matriz_sim.min(), matriz_sim.max())

    return matriz_sim, cvegeo




# ============================================================
# EJEMPLO: LLAMADA PRINCIPAL (puedes editar este bloque)
# ============================================================
if __name__ == "__main__":

    # Rutas de tus matrices individuales
    paths = [
        "outputs/matriz_categorica.h5",
        "outputs/matriz_numerica.h5",
        "outputs/matriz_sequia.h5"
    ]

    # Pesos para cada matriz (puedes cambiarlos)


    # Archivo de salida
    salida = "outputs/matriz_similitud_general.h5"

    matriz, cvegeo = matriz_similitud_general(paths, salida)
