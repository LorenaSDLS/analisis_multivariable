#normalizar
import time

def normalizar_cvegeo(cve):
    return str(cve).zfill(5)



class Progreso:
    """
    Clase ligera para imprimir progreso periódicamente con ETA.
    """

    def __init__(self, total, cada=100, etiqueta="Progreso"):
        self.total = total
        self.cada = cada
        self.etiqueta = etiqueta
        self.inicio = time.time()
        self.ultimo = 0

    def paso(self, i):
        """
        Llamar a esta función en cada iteración del loop principal.
        """
        if i - self.ultimo < self.cada:
            return
        self.ultimo = i

        elapsed = time.time() - self.inicio
        pct = i / self.total
        eta = elapsed / pct - elapsed if pct > 0 else 0

        print(
            f"[{self.etiqueta}] {pct:5.1%} | fila {i}/{self.total} | "
            f"ETA: {eta/60:.1f} min"
        )
