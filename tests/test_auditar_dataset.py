"""Auditoria da amostra usa apenas a biblioteca padrão, sem pesos nem rede."""

import tempfile
import unittest
from pathlib import Path

from auditar_dataset import auditar, validar_linha


class TestValidarLinha(unittest.TestCase):
    def test_linha_correta(self):
        self.assertEqual(validar_linha("2 0.5 0.5 1 1", "a.txt", 1), 2)

    def test_classe_fora_do_limite(self):
        with self.assertRaisesRegex(ValueError, "classe"):
            validar_linha("4 0.5 0.5 0.1 0.2", "a.txt", 1)

    def test_coordenadas_invalidas(self):
        for linha in ("1 0.5 0.5 0 0.3", "0 0.95 0.5 0.5 0.2",
                      "1 0.3 1.5 0.3 0.3", "0 a b c d"):
            with self.subTest(linha=linha), self.assertRaises(ValueError):
                validar_linha(linha, "a.txt", 1)

    def test_formato(self):
        with self.assertRaisesRegex(ValueError, "formato"):
            validar_linha("0 0.2 0.3", "a.txt", 1)


class TestAuditoria(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for split in ("train", "valid", "test"):
            (self.root / split / "images").mkdir(parents=True)
            (self.root / split / "labels").mkdir(parents=True)

    def adicionar(self, split, nome, conteudo):
        (self.root / split / "images" / f"{nome}.jpg").write_bytes(conteudo)
        (self.root / split / "labels" / f"{nome}.txt").write_text(
            "0 0.5 0.5 0.25 0.25\n", encoding="utf-8"
        )

    def test_amostra_valida(self):
        for i, split in enumerate(("train", "valid", "test")):
            self.adicionar(split, f"imagem_{i}", f"diferente_{i}".encode())
        res = auditar(self.root)
        self.assertTrue(res["aprovado"])
        self.assertEqual(res["divisoes"]["test"]["imagens"], 1)
        self.assertEqual(res["divisoes"]["test"]["por_classe"]["bolsa"], 1)

    def test_deduplicacao_exata_entre_splits(self):
        self.adicionar("train", "a", b"repetida")
        self.adicionar("test", "b", b"repetida")
        res = auditar(self.root)
        self.assertFalse(res["aprovado"])
        self.assertTrue(res["duplicados_entre_divisoes"])

    def test_rotulo_ausente(self):
        (self.root / "test/images" / "sem_label.jpg").write_bytes(b"imagem")
        res = auditar(self.root)
        self.assertTrue(any("rótulo ausente" in e for e in res["erros"]))

    def test_rotulo_orfao(self):
        (self.root / "test/labels" / "orfao.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
        )
        res = auditar(self.root)
        self.assertTrue(any("rótulo sem imagem" in e for e in res["erros"]))


if __name__ == "__main__":
    unittest.main()
