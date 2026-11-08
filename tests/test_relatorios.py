"""Testa CSV/JSON, classes, coordenadas e zero detecções."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relatorios import COLUNAS, extrair_deteccoes, salvar_relatorios
from tests.helpers import ResultadoFalso


class TestRelatorios(unittest.TestCase):
    def test_extrai_classe_confianca_coordenadas_e_quadro(self):
        resultado = ResultadoFalso([(1, 0.876543219, [10.123, 20, 30, 40])])
        linhas = extrair_deteccoes(resultado, "imagem.jpg", quadro=2, tempo_s=0.04)
        self.assertEqual(linhas[0]["classe"], "oculos")
        self.assertEqual(linhas[0]["classe_id"], 1)
        self.assertEqual(linhas[0]["confianca"], 0.876543)
        self.assertEqual(linhas[0]["x1"], 10.12)
        self.assertEqual(linhas[0]["x2"], 30.0)
        self.assertEqual(linhas[0]["quadro"], 2)
        self.assertEqual(linhas[0]["tempo_s"], 0.04)

    def test_nomes_em_lista(self):
        resultado = ResultadoFalso([(0, 0.4, [1, 2, 3, 4])],
                                   nomes=["relogio"])
        self.assertEqual(extrair_deteccoes(resultado, "a.jpg")[0]["classe"], "relogio")

    def test_classe_desconhecida_tem_fallback(self):
        resultado = ResultadoFalso([(10, 0.4, [1, 2, 3, 4])])
        self.assertEqual(extrair_deteccoes(resultado, "a.jpg")[0]["classe"], "classe_10")

    def test_sem_deteccoes(self):
        self.assertEqual(extrair_deteccoes(ResultadoFalso(), "vazio.jpg"), [])

    def test_csv_com_cabecalho_e_json_vazio(self):
        with tempfile.TemporaryDirectory() as pasta:
            destino = Path(pasta)
            salvar_relatorios(destino, [], {"total_deteccoes": 0})
            with (destino / "deteccoes.csv").open(encoding="utf-8-sig", newline="") as f:
                leitor = csv.DictReader(f)
                self.assertEqual(leitor.fieldnames, list(COLUNAS))
                self.assertEqual(list(leitor), [])
            dados = json.loads((destino / "resumo.json").read_text(encoding="utf-8"))
            self.assertEqual(dados["deteccoes"], [])
            self.assertEqual(dados["resumo"]["total_deteccoes"], 0)

    def test_roundtrip_csv_json(self):
        with tempfile.TemporaryDirectory() as pasta:
            destino = Path(pasta)
            linhas = extrair_deteccoes(
                ResultadoFalso([(0, 0.95, [4, 5, 8, 9])]), "foto.jpg"
            )
            salvar_relatorios(destino, linhas, {"total_imagens": 1})
            with (destino / "deteccoes.csv").open(encoding="utf-8-sig", newline="") as f:
                registros = list(csv.DictReader(f))
            self.assertEqual(registros[0]["arquivo"], "foto.jpg")
            self.assertEqual(float(registros[0]["confianca"]), 0.95)
            dados = json.loads((destino / "resumo.json").read_text(encoding="utf-8"))
            self.assertEqual(dados["deteccoes"], linhas)


if __name__ == "__main__":
    unittest.main()
