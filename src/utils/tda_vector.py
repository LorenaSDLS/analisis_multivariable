# src/utils/tda_vector.py
from gtda.diagrams import PersistenceEntropy, PersistenceImager  # o PersistenceImage en tu versión
from joblib import Parallel, delayed
import numpy as np

def diagramas_a_images(lista_diagramas, n_jobs=-1, image_params=None):
    # usa PersistenceImager si tu versión la tiene; en giotto-tda hay PersistenceImager o similar
    pi = PersistenceImager(**(image_params or {}))
    X = pi.fit_transform(lista_diagramas)  # devuelve shape (n_samples, h, w) or flattened
    # aplana si hace falta
    X_flat = X.reshape(X.shape[0], -1)
    return X_flat
