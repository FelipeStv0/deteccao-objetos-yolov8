"""Objetos simulados para testar resultados YOLO sem instalar PyTorch."""


class TensorFalso:
    def __init__(self, valores):
        self.valores = valores

    def tolist(self):
        return self.valores


class CaixasFalsas:
    def __init__(self, linhas=()):
        # linhas: (classe_id, confianca, (x1,y1,x2,y2))
        self.linhas = list(linhas)
        self.cls = TensorFalso([x[0] for x in self.linhas])
        self.conf = TensorFalso([x[1] for x in self.linhas])
        self.xyxy = TensorFalso([x[2] for x in self.linhas])

    def __len__(self):
        return len(self.linhas)


class ResultadoFalso:
    def __init__(self, linhas=(), nomes=None):
        self.boxes = CaixasFalsas(linhas)
        self.names = nomes if nomes is not None else {0: "bolsa", 1: "oculos"}

    def plot(self):
        return "quadro_bgr_anotado"
