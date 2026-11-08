"""Testes do invólucro de inferência, sem carregar pesos nem utilizar rede."""

import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import inferir


class TestListarImagens(unittest.TestCase):
    def test_lista_apenas_imagens_em_ordem(self):
        with tempfile.TemporaryDirectory() as pasta:
            raiz = Path(pasta)
            for nome in ["z.PNG", "a.jpg", "anotacao.txt"]:
                (raiz / nome).touch()
            (raiz / "subpasta").mkdir()
            (raiz / "subpasta" / "oculta.jpg").touch()
            self.assertEqual([p.name for p in inferir.listar_imagens(raiz)],
                             ["a.jpg", "z.PNG"])

    def test_arquivo_individual(self):
        with tempfile.TemporaryDirectory() as pasta:
            imagem = Path(pasta) / "foto.JPEG"
            imagem.touch()
            self.assertEqual(inferir.listar_imagens(imagem), [imagem])

    def test_entrada_ausente(self):
        with self.assertRaises(FileNotFoundError):
            inferir.listar_imagens(Path("imagem_que_nao_existe_123.jpg"))

    def test_pasta_sem_imagens(self):
        with tempfile.TemporaryDirectory() as pasta:
            with self.assertRaisesRegex(ValueError, "Nenhuma imagem"):
                inferir.listar_imagens(Path(pasta))

    def test_formato_invalido(self):
        with tempfile.TemporaryDirectory() as pasta:
            arquivo = Path(pasta) / "video.mp4"
            arquivo.touch()
            with self.assertRaisesRegex(ValueError, "Formato"):
                inferir.listar_imagens(arquivo)


class TestExecutar(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.raiz = Path(self.temp.name)
        self.pesos = self.raiz / "best.pt"
        self.pesos.touch()
        self.pasta_imagens = self.raiz / "imagens"
        self.pasta_imagens.mkdir()
        for nome in ("a.jpg", "b.png"):
            (self.pasta_imagens / nome).touch()
        self.saida = self.raiz / "resultados"

    def test_predicao_usa_pesos_locais_e_salva_resultados(self):
        chamadas = []
        carregamentos = []

        class ModeloFalso:
            def __init__(self, caminho):
                carregamentos.append(caminho)

            def predict(self, **kwargs):
                chamadas.append(kwargs)
                return [SimpleNamespace(boxes=[object(), object()])]

        with patch.dict(sys.modules, {"ultralytics": SimpleNamespace(YOLO=ModeloFalso)}):
            with patch("sys.stdout", new_callable=io.StringIO):
                self.assertEqual(inferir.executar(
                    self.pasta_imagens, self.pesos, self.saida, 0.3, 640, "cpu"
                ), (2, 4))

        self.assertEqual(carregamentos, [str(self.pesos)])
        self.assertEqual([Path(c["source"]).name for c in chamadas],
                         ["a.jpg", "b.png"])
        self.assertTrue(all(c["save"] and c["exist_ok"] for c in chamadas))
        self.assertTrue(all(c["conf"] == 0.3 and c["imgsz"] == 640 for c in chamadas))
        self.assertTrue(all(c["project"] == str(self.saida.resolve().parent)
                            and c["name"] == self.saida.name for c in chamadas))

    def test_pesos_ausentes(self):
        with self.assertRaisesRegex(FileNotFoundError, "Pesos"):
            inferir.executar(self.pasta_imagens, self.raiz / "nao.pt",
                             self.saida, 0.25, 640, "cpu")

    def test_conf_invalida(self):
        for conf in (-0.1, 1.1):
            with self.subTest(conf=conf), self.assertRaisesRegex(ValueError, "conf"):
                inferir.executar(self.pasta_imagens, self.pesos,
                                 self.saida, conf, 640, "cpu")

    def test_saida_na_mesma_pasta_e_rejeitada(self):
        with self.assertRaisesRegex(ValueError, "saída"):
            inferir.executar(self.pasta_imagens, self.pesos,
                             self.pasta_imagens, 0.25, 640, "cpu")

    def test_imgsz_invalido(self):
        with self.assertRaisesRegex(ValueError, "imgsz"):
            inferir.executar(self.pasta_imagens, self.pesos,
                             self.saida, 0.25, 0, "cpu")

    def test_main_erro_retorna_codigo_um(self):
        with patch("sys.stderr", new_callable=io.StringIO) as err:
            codigo = inferir.main(["--input", str(self.pasta_imagens),
                                   "--model", str(self.raiz / "inexistente.pt")])
        self.assertEqual(codigo, 1)
        self.assertIn("Pesos não encontrados", err.getvalue())


if __name__ == "__main__":
    unittest.main()
