"""Contratos públicos dos novos módulos, sem dependências de ML."""
import tempfile
import unittest
from pathlib import Path

import entrada_imagens
import inferir
from validacao import validar_parametros, validar_pesos


class TestModulos(unittest.TestCase):
    def test_lista_imagens_continua_compativel(self):
        self.assertIs(inferir.listar_imagens, entrada_imagens.listar_imagens)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "b.PNG").touch()
            (root / "a.jpg").touch()
            (root / "notas.txt").touch()
            self.assertEqual([p.name for p in entrada_imagens.listar_imagens(root)],
                             ["a.jpg", "b.PNG"])

    def test_parametros_invalidos(self):
        for value in (-1, 1.01, float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validar_parametros(value, 640)
        with self.assertRaisesRegex(ValueError, "imgsz"):
            validar_parametros(0.25, 0)

    def test_checkpoint_inexistente(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                validar_pesos(Path(directory) / "missing.pt")


if __name__ == "__main__":
    unittest.main()
