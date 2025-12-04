# src/construccion_matriz/matriz_blocks.py

import numpy as np
from multiprocessing import Pool, cpu_count


# ======================================
#   DISTANCIA L2 ENTRE SERIES
# ======================================

def distancia_l2(a, b):
    """Distancia L2 entre dos arrays numpy del mismo tamaño."""
    diff = a - b
    return float(np.sqrt(np.sum(diff * diff)))


# ======================================
#   DISTANCIA → SIMILITUD (0–1)
# ======================================

def distancia_a_similitud(d):
    """Convierte distancia L2 a similitud 0–1."""
    return 1.0 / (1.0 + float(d))


# ======================================
#   COMPARACIÓN BATCH DE UN BLOQUE
# ======================================

def comparar_bloque(args):
    """
    Compara un bloque de filas A contra TODAS las filas B.
    """
    A_block, B_full = args
    nA = A_block.shape[0]
    nB = B_full.shape[0]

    resultado = np.zeros((nA, nB), dtype=np.float32)

    for i in range(nA):
        for j in range(nB):
            d = distancia_l2(A_block[i], B_full[j])
            resultado[i, j] = distancia_a_similitud(d)

    return resultado


# ======================================
#   MATRIZ COMPLETA POR BLOQUES
# ======================================

def construir_matriz_similitud_blocks(df_wide, lista_mun, block_size=300,
                                      n_procs=4, use_parallel=True):

    N = len(lista_mun)
    columnas = df_wide.shape[1]

    print(f"Construyendo matriz por bloques: {N}x{N}")

    # Convertimos a matriz numpy completa
    M = df_wide.to_numpy(dtype=float)

    # Matriz de salida
    out = np.zeros((N, N), dtype=np.float32)

    # Preparamos bloques
    bloques = []
    for i in range(0, N, block_size):
        A = M[i:i + block_size]
        bloques.append((i, A))

    # Procesamiento
    if use_parallel:
        n_procs = min(n_procs, cpu_count())
        print(f"Procesando en paralelo con {n_procs} procesos...")

        with Pool(processes=n_procs) as pool:
            for (idxA, A_block), resultado in zip(
                bloques,
                pool.imap(comparar_bloque, [(A_block, M) for (_, A_block) in bloques])
            ):
                nA = A_block.shape[0]
                out[idxA:idxA + nA, :] = resultado

    else:
        print("Procesando en modo secuencial...")
        for idxA, A_block in bloques:
            resultado = comparar_bloque((A_block, M))
            nA = A_block.shape[0]
            out[idxA:idxA + nA, :] = resultado

    return lista_mun, out
