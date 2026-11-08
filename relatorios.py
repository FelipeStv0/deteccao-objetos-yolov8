"""Exportação determinística das predições (não é avaliação de precisão)."""

import csv
import json
from pathlib import Path

COLUNAS = (
    "arquivo", "quadro", "tempo_s", "classe_id", "classe",
    "confianca", "x1", "y1", "x2", "y2",
)


def extrair_deteccoes(resultado, arquivo: str, quadro: int | None = None,
                      tempo_s: float | None = None) -> list[dict]:
    """Extrai caixas xyxy em pixels dos Results do Ultralytics."""
    caixas = resultado.boxes
    if caixas is None or len(caixas) == 0:
        return []

    nomes = resultado.names
    linhas = []
    for coordenadas, classe, confianca in zip(
        caixas.xyxy.tolist(), caixas.cls.tolist(), caixas.conf.tolist()
    ):
        classe_id = int(classe)
        if isinstance(nomes, dict):
            rotulo = nomes.get(classe_id, f"classe_{classe_id}")
        elif 0 <= classe_id < len(nomes):
            rotulo = nomes[classe_id]
        else:
            rotulo = f"classe_{classe_id}"

        linhas.append({
            "arquivo": arquivo,
            "quadro": quadro,
            "tempo_s": tempo_s,
            "classe_id": classe_id,
            "classe": str(rotulo),
            "confianca": round(float(confianca), 6),
            "x1": round(float(coordenadas[0]), 2),
            "y1": round(float(coordenadas[1]), 2),
            "x2": round(float(coordenadas[2]), 2),
            "y2": round(float(coordenadas[3]), 2),
        })
    return linhas


def salvar_relatorios(pasta: Path, deteccoes: list[dict], resumo: dict) -> None:
    """Escreve CSV (inclusive vazio, com cabeçalho) e JSON legíveis."""
    pasta.mkdir(parents=True, exist_ok=True)
    with (pasta / "deteccoes.csv").open("w", newline="", encoding="utf-8-sig") as f:
        escritor = csv.DictWriter(f, fieldnames=COLUNAS)
        escritor.writeheader()
        escritor.writerows(deteccoes)

    with (pasta / "resumo.json").open("w", encoding="utf-8") as f:
        json.dump({"resumo": resumo, "deteccoes": deteccoes},
                  f, ensure_ascii=False, indent=2)
        f.write("\n")
