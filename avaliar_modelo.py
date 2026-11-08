"""Avaliação EXPLORATÓRIA da amostra rotulada, erros e benchmark em CPU/GPU.

Os 10 exemplos de teste NÃO representam o dataset completo; os resultados
não são comparáveis às métricas documentadas para o treinamento original.
"""

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

from auditar_dataset import CLASSES, ROOT, auditar

COLUNAS_ERRO = ("imagem", "tipo", "classe_id", "classe", "confianca",
                "iou", "x1", "y1", "x2", "y2")


def normalizar_nomes(nomes) -> dict[int, str]:
    """Converte lista ou dicionário de classes em IDs inteiros ordenados."""
    if isinstance(nomes, dict):
        classes = {int(i): str(nome) for i, nome in nomes.items()}
    else:
        classes = dict(enumerate(map(str, nomes)))
    if sorted(classes) != list(range(len(classes))):
        raise ValueError("IDs das classes do checkpoint não são contíguos")
    return classes


def extrair_metricas(resultado, nomes: dict[int, str]) -> dict:
    """Extrai métricas usando ap_class_index (e não um índice ingênuo)."""
    box = resultado.box
    geral = {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "map50": float(box.map50),
        "map50_95": float(box.map),
    }
    por_classe = []
    indices = [int(i) for i in box.ap_class_index]
    for indice_local, id_classe in enumerate(indices):
        por_classe.append({
            "classe_id": id_classe,
            "classe": nomes[id_classe],
            "precision": float(box.p[indice_local]),
            "recall": float(box.r[indice_local]),
            "map50": float(box.ap50[indice_local]),
            "map50_95": float(box.ap[indice_local]),
        })
    return {"geral": geral, "por_classe": por_classe}


def iou_xyxy(a, b) -> float:
    """Interseção/sobreposição de duas caixas em pixels, entre 0 e 1."""
    xi = max(a[0], b[0]), max(a[1], b[1])
    xf = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xf[0] - xi[0]) * max(0, xf[1] - xi[1])
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    uniao = area_a + area_b - inter
    return inter / uniao if uniao > 0 else 0.0


def ler_ground_truth(path: Path, largura: int, altura: int) -> list[dict]:
    """Lê classe e coordenadas YOLO normalizadas de um arquivo validado."""
    gts = []
    for linha in path.read_text(encoding="utf-8").splitlines():
        if not linha.strip():
            continue
        cls, cx, cy, w, h = map(float, linha.split())
        gts.append({
            "classe_id": int(cls),
            "xyxy": [(cx - w / 2) * largura, (cy - h / 2) * altura,
                     (cx + w / 2) * largura, (cy + h / 2) * altura],
        })
    return gts


def identificar_erros(gts: list[dict], preds: list[dict],
                      imagem: str, nomes: dict[int, str], limiar_iou=0.5):
    """Pareamento greedy por maior IoU dentro da mesma classe (diagnóstico)."""
    candidatos = []
    for g, gt in enumerate(gts):
        for p, pred in enumerate(preds):
            if gt["classe_id"] == pred["classe_id"]:
                iou = iou_xyxy(gt["xyxy"], pred["xyxy"])
                if iou >= limiar_iou:
                    candidatos.append((iou, g, p))
    usados_gt, usados_pred = set(), set()
    por_classe = Counter()
    for _, g, p in sorted(candidatos, reverse=True):
        if g not in usados_gt and p not in usados_pred:
            usados_gt.add(g)
            usados_pred.add(p)
            por_classe[(gts[g]["classe_id"], "tp")] += 1

    erros = []
    for indice, pred in enumerate(preds):
        if indice in usados_pred:
            continue
        cls = pred["classe_id"]
        por_classe[(cls, "fp")] += 1
        erros.append({
            "imagem": imagem, "tipo": "falso_positivo",
            "classe_id": cls, "classe": nomes[cls],
            "confianca": round(pred["conf"], 6), "iou": "",
            **dict(zip(("x1", "y1", "x2", "y2"),
                       (round(v, 2) for v in pred["xyxy"]))),
        })
    for indice, gt in enumerate(gts):
        if indice in usados_gt:
            continue
        cls = gt["classe_id"]
        por_classe[(cls, "fn")] += 1
        erros.append({
            "imagem": imagem, "tipo": "falso_negativo",
            "classe_id": cls, "classe": nomes[cls],
            "confianca": "", "iou": "",
            **dict(zip(("x1", "y1", "x2", "y2"),
                       (round(v, 2) for v in gt["xyxy"]))),
        })
    return erros, por_classe


def resumir_erros(totais: Counter, nomes: dict[int, str]) -> list[dict]:
    relatorio = []
    for id_classe, nome in nomes.items():
        tp = totais[(id_classe, "tp")]
        fp = totais[(id_classe, "fp")]
        fn = totais[(id_classe, "fn")]
        relatorio.append({
            "classe_id": id_classe, "classe": nome, "tp": tp, "fp": fp, "fn": fn,
            "precision_ponto_operacao": tp / (tp + fp) if tp + fp else None,
            "recall_ponto_operacao": tp / (tp + fn) if tp + fn else None,
        })
    return relatorio


def percentil_nearest_rank(valores: list[float], percentil: float) -> float:
    if not valores:
        raise ValueError("Não há medições disponíveis")
    ordenados = sorted(valores)
    indice = max(0, math.ceil(len(ordenados) * percentil) - 1)
    return ordenados[indice]


def benchmark(modelo, entrada: Path, dispositivo: str, imgsz: int,
              aquecimentos: int, repeticoes: int) -> dict:
    """Latência wall-clock por imagem, após carregar o checkpoint.

    Inclui carregar arquivo da imagem e pré/pós-processar, mas não criar o
    modelo, gerar relatórios ou gravar imagens. Não representa FPS de vídeo.
    """
    if aquecimentos < 0 or repeticoes < 1:
        raise ValueError("Aquecimentos >= 0 e repetições >= 1 são obrigatórios")
    tempos_ms = []
    for indice in range(aquecimentos + repeticoes):
        inicio = time.perf_counter()
        modelo.predict(source=str(entrada), device=dispositivo,
                       imgsz=imgsz, conf=0.25, verbose=False)
        duracao = (time.perf_counter() - inicio) * 1000
        if indice >= aquecimentos:
            tempos_ms.append(duracao)

    media = statistics.mean(tempos_ms)
    return {
        "imagem": entrada.name,
        "aquecimentos": aquecimentos,
        "repeticoes": repeticoes,
        "latencia_ms": {
            "media": round(media, 3),
            "mediana": round(statistics.median(tempos_ms), 3),
            "p95_nearest_rank": round(percentil_nearest_rank(tempos_ms, 0.95), 3),
            "min": round(min(tempos_ms), 3),
            "max": round(max(tempos_ms), 3),
        },
        "taxa_teorica_imagens_por_s": round(1000 / media, 3) if media else None,
        "escopo": "predict de imagem única, modelo carregado; inclui leitura, pré/pós-processamento; sem gravação",
    }


def executar(base: Path, pesos: Path, saida: Path, dispositivo: str = "cpu",
             imgsz: int = 640, conf: float = 0.25, aquecimentos: int = 3,
             repeticoes: int = 15) -> dict:
    if not pesos.is_file():
        raise FileNotFoundError(f"Checkpoint não encontrado: {pesos}")
    if imgsz <= 0 or not 0 <= conf <= 1:
        raise ValueError("imgsz deve ser positivo e conf deve estar entre 0 e 1")
    auditoria = auditar(base)
    if not auditoria["aprovado"]:
        raise ValueError("Auditoria da amostra falhou: " + "; ".join(auditoria["erros"]))
    saida = saida.resolve()
    saida.mkdir(parents=True, exist_ok=True)
    (saida / "auditoria_dataset.json").write_text(
        json.dumps(auditoria, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    from ultralytics import YOLO, __version__ as ultralytics_version
    import torch
    import yaml

    modelo = YOLO(str(pesos.resolve()))
    nomes = normalizar_nomes(modelo.names)
    if len(nomes) != len(CLASSES):
        raise ValueError(
            f"Checkpoint possui {len(nomes)} classes, rótulos esperam {len(CLASSES)}"
        )

    # YAML criado com caminho ABSOLUTO para evitar dependência do diretório
    # de execução e nomes extraídos do checkpoint, não inventados.
    config = {
        "path": str(base.resolve()), "train": "train/images",
        "val": "valid/images", "test": "test/images", "names": nomes,
    }
    config_path = saida / "dataset_avaliado.yaml"
    config_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    metricas_ultralytics = modelo.val(
        data=str(config_path), split="test", device=dispositivo,
        imgsz=imgsz, conf=0.001, plots=False,
        project=str(saida), name="validacao_ultralytics",
        exist_ok=True, verbose=False,
    )
    metricas = extrair_metricas(metricas_ultralytics, nomes)
    imagens = sorted((base / "test/images").glob("*"))
    imagens = [x for x in imagens if x.suffix.lower()
               in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}]
    erros = []
    totais = Counter()
    por_imagem = []
    for imagem in imagens:
        resultado = modelo.predict(
            source=str(imagem), device=dispositivo,
            imgsz=imgsz, conf=conf, verbose=False,
        )[0]
        altura, largura = resultado.orig_shape
        gts = ler_ground_truth(
            base / "test/labels" / (imagem.stem + ".txt"),
            largura, altura,
        )
        caixas = resultado.boxes
        preds = []
        if caixas is not None and len(caixas):
            preds = [
                {"classe_id": int(cls), "conf": float(c),
                 "xyxy": [float(z) for z in xyxy]}
                for xyxy, cls, c in zip(
                    caixas.xyxy.tolist(), caixas.cls.tolist(), caixas.conf.tolist()
                )
            ]
        encontrados, contagem = identificar_erros(gts, preds, imagem.name, nomes)
        erros.extend(encontrados)
        totais.update(contagem)
        por_imagem.append({
            "imagem": imagem.name, "ground_truth": len(gts),
            "predicoes": len(preds),
            "falsos_positivos": sum(v for (c, k), v in contagem.items() if k == "fp"),
            "falsos_negativos": sum(v for (c, k), v in contagem.items() if k == "fn"),
        })

    with (saida / "erros.csv").open("w", newline="", encoding="utf-8-sig") as f:
        escritor = csv.DictWriter(f, fieldnames=COLUNAS_ERRO)
        escritor.writeheader()
        escritor.writerows(erros)

    metadados = {
        "status": "exploratorio_apenas_amostra",
        "amostra_imagens_teste": len(imagens),
        "classes_checkpoint": nomes,
        "modelo_sha256": hashlib.sha256(pesos.read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "plataforma": platform.platform(),
        "pytorch": torch.__version__,
        "ultralytics": ultralytics_version,
        "dispositivo_solicitado": dispositivo,
        "imgsz": imgsz, "conf_diagnostico": conf,
        "limiar_iou_diagnostico": 0.5,
        "observacao": "Amostra pequena, originalmente derivada de dataset maior; não validar generalização.",
    }
    relatorio = {
        "metadados": metadados,
        "metricas_ultralytics_amostra": metricas,
        "diagnostico_operacional": {
            "por_classe": resumir_erros(totais, nomes),
            "por_imagem": por_imagem,
            "total_erros": len(erros),
            "metodo": "pareamento greedy mesma classe / IoU >= 0.5 / confiança fixa",
        },
    }
    (saida / "metricas_amostra.json").write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    bench = benchmark(
        modelo, imagens[0], dispositivo, imgsz, aquecimentos, repeticoes
    )
    bench["metadados"] = metadados
    (saida / "benchmark.json").write_text(
        json.dumps(bench, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "imagens": len(imagens),
        "erros": len(erros),
        "map50": metricas["geral"]["map50"],
        "map50_95": metricas["geral"]["map50_95"],
        "latencia_media_ms": bench["latencia_ms"]["media"],
        "saida": str(saida),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Avaliação exploratória e benchmark na amostra anotada YOLO."
    )
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset_amostra")
    parser.add_argument("--model", type=Path, default=ROOT / "modelo/best.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/avaliacao_amostra")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args(argv)
    try:
        resultado = executar(
            args.dataset, args.model, args.output, args.device,
            args.imgsz, args.conf, args.warmup, args.repeats
        )
    except (ValueError, OSError, RuntimeError) as erro:
        print(f"Erro: {erro}", file=sys.stderr)
        return 1
    print(json.dumps(resultado, ensure_ascii=False, indent=2))
    print("AVISO: métricas referentes SOMENTE às 10 imagens de teste da amostra.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
