"""Testes matemáticos do diagnóstico e benchmark, sem Ultralytics."""

import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from avaliar_modelo import (benchmark, extrair_metricas, identificar_erros,
                           iou_xyxy, normalizar_nomes, percentil_nearest_rank,
                           salvar_comparacao,
                           resumir_erros)


class TestAvaliacao(unittest.TestCase):
    def test_iou(self):
        self.assertAlmostEqual(iou_xyxy([0, 0, 2, 2], [1, 1, 3, 3]), 1 / 7)
        self.assertEqual(iou_xyxy([0, 0, 1, 1], [2, 2, 3, 3]), 0.0)

    def test_nomes(self):
        self.assertEqual(normalizar_nomes({"0": "bolsa", "1": "oculos"}),
                         {0: "bolsa", 1: "oculos"})
        self.assertEqual(normalizar_nomes(["bolsa", "oculos"]),
                         {0: "bolsa", 1: "oculos"})
        with self.assertRaises(ValueError):
            normalizar_nomes({0: "bolsa", 2: "relogio"})

    def test_pareamento_mesma_classe(self):
        gts = [{"classe_id": 0, "xyxy": [0, 0, 10, 10]},
               {"classe_id": 1, "xyxy": [20, 20, 30, 30]}]
        preds = [
            {"classe_id": 0, "conf": 0.9, "xyxy": [0, 0, 10, 10]},
            {"classe_id": 0, "conf": 0.7, "xyxy": [20, 20, 30, 30]},
        ]
        erros, totais = identificar_erros(gts, preds, "a.jpg",
                                          {0: "bolsa", 1: "oculos"})
        self.assertEqual(totais[(0, "tp")], 1)
        self.assertEqual(totais[(0, "fp")], 1)
        self.assertEqual(totais[(1, "fn")], 1)
        self.assertEqual({e["tipo"] for e in erros},
                         {"falso_positivo", "falso_negativo"})

    def test_sem_caixas(self):
        erros, totais = identificar_erros([], [], "a.jpg", {0: "bolsa"})
        self.assertEqual(erros, [])
        self.assertFalse(totais)

    def test_precision_recall_ponto_operacao(self):
        data = Counter({(0, "tp"): 2, (0, "fp"): 1, (0, "fn"): 2})
        item = resumir_erros(data, {0: "bolsa"})[0]
        self.assertAlmostEqual(item["precision_ponto_operacao"], 2 / 3)
        self.assertAlmostEqual(item["recall_ponto_operacao"], 0.5)

    def test_map_indexado_por_ap_class_index(self):
        metric = SimpleNamespace(
            box=SimpleNamespace(
                mp=0.4, mr=0.5, map50=0.6, map=0.3,
                ap_class_index=[2, 0], p=[0.2, 0.8], r=[0.1, 0.9],
                ap50=[0.4, 0.7], ap=[0.25, 0.55],
            )
        )
        resumo = extrair_metricas(metric, {0: "bolsa", 1: "oculos", 2: "relogio"})
        self.assertEqual(resumo["por_classe"][0]["classe"], "relogio")
        self.assertEqual(resumo["por_classe"][1]["classe"], "bolsa")
        self.assertEqual(resumo["geral"]["map50_95"], 0.3)

    def test_percentil_nearest_rank(self):
        self.assertEqual(percentil_nearest_rank(list(range(1, 21)), 0.95), 19)
        with self.assertRaises(ValueError):
            percentil_nearest_rank([], 0.95)

    def test_benchmark_repeticoes(self):
        class Modelo:
            def __init__(self):
                self.calls = []
            def predict(self, **kwargs):
                self.calls.append(kwargs)
        modelo = Modelo()
        with patch("avaliar_modelo.time.perf_counter",
                   side_effect=[0, 0.1, 1, 1.2, 2, 2.3]):
            resultado = benchmark(modelo, Path("img.jpg"), "cpu", 640, 1, 2)
        self.assertEqual(len(modelo.calls), 3)
        self.assertEqual(resultado["repeticoes"], 2)
        self.assertAlmostEqual(resultado["latencia_ms"]["media"], 250.0)
        self.assertEqual(resultado["latencia_ms"]["p95_nearest_rank"], 300.0)


    def test_comparacao_desenha_gt_e_predicoes(self):
        chamadas = {"retangulos": 0, "textos": []}
        cv2_falso = SimpleNamespace(
            rectangle=lambda *a: chamadas.__setitem__(
                "retangulos", chamadas["retangulos"] + 1
            ),
            putText=lambda canvas, texto, *a: chamadas["textos"].append(texto),
            imwrite=lambda *a: True,
            FONT_HERSHEY_SIMPLEX=0,
        )
        imagem_falsa = SimpleNamespace(copy=lambda: object())
        with tempfile.TemporaryDirectory() as pasta:
            with patch.dict(sys.modules, {"cv2": cv2_falso}):
                salvar_comparacao(
                    imagem_falsa,
                    [{"classe_id": 0, "xyxy": [1, 2, 12, 15]}],
                    [{"classe_id": 0, "conf": 0.93, "xyxy": [1, 2, 12, 15]}],
                    {0: "bolsa"},
                    Path(pasta) / "comparacoes" / "a.jpg",
                )
        self.assertEqual(chamadas["retangulos"], 2)
        self.assertTrue(chamadas["textos"][0].startswith("GT bolsa"))
        self.assertTrue(chamadas["textos"][1].startswith("PRED bolsa"))



if __name__ == "__main__":
    unittest.main()
