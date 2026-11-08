"""Testes de vídeo com captura/escrita e pesos simulados (sem OpenCV)."""

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import inferir_video
from tests.helpers import ResultadoFalso


class TestVideo(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.raiz = Path(self.temp.name)
        self.video = self.raiz / "video_original.mp4"
        self.video.touch()
        self.pesos = self.raiz / "best.pt"
        self.pesos.touch()
        self.saida = self.raiz / "runs" / "video_anotado.mp4"

    def test_video_processa_quadros_e_exporta_relatorios(self):
        grava = []
        capturas = []
        gravadores = []
        previsoes = []

        class Captura:
            def __init__(self, caminho):
                self.caminho = caminho
                self.indice = 0
                self.liberado = False
                capturas.append(self)

            def isOpened(self):
                return True

            def get(self, propriedade):
                return {1: 25.0, 2: 128, 3: 72}[propriedade]

            def read(self):
                self.indice += 1
                return (True, f"frame_{self.indice}") if self.indice <= 3 else (False, None)

            def release(self):
                self.liberado = True

        class Gravador:
            def __init__(self, *args):
                self.args = args
                self.liberado = False
                gravadores.append(self)

            def isOpened(self):
                return True

            def write(self, frame):
                grava.append(frame)

            def release(self):
                self.liberado = True

        class Modelo:
            def __init__(self, caminho):
                self.caminho = caminho

            def predict(self, **kwargs):
                previsoes.append(kwargs)
                return [ResultadoFalso([(1, 0.93, [5, 10, 20, 40])])]

        cv2_falso = SimpleNamespace(
            VideoCapture=Captura, VideoWriter=Gravador,
            VideoWriter_fourcc=lambda *args: 123,
            CAP_PROP_FPS=1, CAP_PROP_FRAME_WIDTH=2, CAP_PROP_FRAME_HEIGHT=3,
        )
        with patch.dict(sys.modules, {
            "cv2": cv2_falso, "ultralytics": SimpleNamespace(YOLO=Modelo)
        }):
            with patch("sys.stdout", new_callable=io.StringIO):
                self.assertEqual(inferir_video.executar_video(
                    self.video, self.pesos, self.saida, 0.25, 640, "cpu", 2
                ), (2, 2))

        self.assertTrue(capturas[0].liberado)
        self.assertTrue(gravadores[0].liberado)
        self.assertEqual(gravadores[0].args[3], (128, 72))
        self.assertEqual(grava, ["quadro_bgr_anotado"] * 2)
        self.assertEqual([p["source"] for p in previsoes], ["frame_1", "frame_2"])
        resumo = json.loads((self.saida.parent / "resumo.json").read_text(encoding="utf-8"))
        self.assertEqual(resumo["resumo"]["quadros_processados"], 2)
        self.assertEqual(resumo["resumo"]["total_deteccoes"], 2)
        self.assertEqual(resumo["deteccoes"][0]["quadro"], 1)
        self.assertEqual(resumo["deteccoes"][1]["tempo_s"], 0.04)

    def test_nao_substitui_video_original(self):
        with self.assertRaisesRegex(ValueError, "substituir"):
            inferir_video.executar_video(
                self.video, self.pesos, self.video, 0.25, 640, "cpu"
            )

    def test_video_ausente(self):
        with self.assertRaisesRegex(FileNotFoundError, "Vídeo não encontrado"):
            inferir_video.executar_video(
                self.raiz / "ausente.mp4", self.pesos, self.saida, 0.25, 640, "cpu"
            )

    def test_extensao_invalida(self):
        outro = self.raiz / "arquivo.txt"
        outro.touch()
        with self.assertRaisesRegex(ValueError, "MP4"):
            inferir_video.executar_video(outro, self.pesos, self.saida, 0.25, 640, "cpu")

    def test_max_frames_negativo(self):
        with self.assertRaisesRegex(ValueError, "max-frames"):
            inferir_video.executar_video(
                self.video, self.pesos, self.saida, 0.25, 640, "cpu", -1
            )

    def test_conf_invalida(self):
        with self.assertRaisesRegex(ValueError, "conf"):
            inferir_video.executar_video(
                self.video, self.pesos, self.saida, 1.1, 640, "cpu"
            )

    def test_saida_nao_mp4(self):
        with self.assertRaisesRegex(ValueError, ".mp4"):
            inferir_video.executar_video(
                self.video, self.pesos, self.raiz / "saida.avi", 0.25, 640, "cpu"
            )

    def test_main_erro_retorna_codigo_um(self):
        with patch("sys.stderr", new_callable=io.StringIO) as err:
            codigo = inferir_video.main(["--input", str(self.raiz / "ausente.mp4")])
        self.assertEqual(codigo, 1)
        self.assertIn("Vídeo não encontrado", err.getvalue())


if __name__ == "__main__":
    unittest.main()
