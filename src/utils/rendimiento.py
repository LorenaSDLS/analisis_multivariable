import time
import psutil
import tracemalloc

# Medición de tiempo
def medir_tiempo(func):
    def wrapper(*args, **kwargs):
        inicio = time.perf_counter()
        resultado = func(*args, **kwargs)
        fin = time.perf_counter()
        return resultado, fin - inicio
    return wrapper

# Medición de memoria (pico)
def iniciar_memoria():
    tracemalloc.start()

def obtener_uso_memoria():
    actual, pico = tracemalloc.get_traced_memory()
    return actual / 1e6, pico / 1e6  # MB

def detener_memoria():
    tracemalloc.stop()

# Medición CPU del proceso
def cpu_actual():
    return psutil.Process().cpu_percent(interval=0.1)
