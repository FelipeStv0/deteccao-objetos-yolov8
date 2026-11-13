"""Detecção em imagens locais com o checkpoint YOLOv8, sem Roboflow."""

import argparse
import sys
from pathlib import Path

from relatorios import extrair_deteccoes, salvar_relatorios

ROOT = Path(__file__).resolve().parent
# Mantém interface anterior para CLI, scripts de terceiros e testes existentes.
from entrada_imagens import EXTENSOES, listar_imagens
from validacao import validar_pesos, validar_parametros


def executar(entrada: Path, pesos: Path, saida: Path, confianca: float,
             tamanho: int, dispositivo: str) -> tuple[int, int]:
    # Contrato comum de validacao para imagens e videos.
    validar_pesos(pesos)
    validar_parametros(confianca, tamanho)

    imagens = listar_imagens(entrada)
    saida = saida.resolve()
    if saida == entrada.resolve() or (
        entrada.is_file() and saida == entrada.parent.resolve()
    ):
        raise ValueError("A pasta de saída não pode substituir as imagens originais")

    # Importação tardia evita exigir PyTorch em testes unitários e --help.
    from ultralytics import YOLO

    modelo = YOLO(str(pesos))
    registros = []
    imagens_resumo = []
    total_deteccoes = 0

    for imagem in imagens:
        resultados = modelo.predict(
            source=str(imagem),
            conf=confianca,
            imgsz=tamanho,
            device=dispositivo,
            save=True,
            project=str(saida.parent),
            name=saida.name,
            exist_ok=True,
            verbose=False,
        )
        deteccoes = [
            linha
            for resultado in resultados
            for linha in extrair_deteccoes(resultado, imagem.name)
        ]
        quantidade = len(deteccoes)
        registros.extend(deteccoes)
        total_deteccoes += quantidade
        imagens_resumo.append({
            "arquivo": imagem.name,
            "deteccoes": quantidade,
            "arquivo_anotado": imagem.name,
        })
        print(f"{imagem.name}: {quantidade} detecção(ões)")

    resumo = {
        "tipo": "imagens",
        "checkpoint": pesos.name,
        "total_imagens": len(imagens),
        "total_deteccoes": total_deteccoes,
        "imagens_sem_deteccoes": sum(
            item["deteccoes"] == 0 for item in imagens_resumo
        ),
        "parametros": {
            "conf": confianca, "imgsz": tamanho, "device": dispositivo,
        },
        "imagens": imagens_resumo,
    }
    salvar_relatorios(saida, registros, resumo)
    return len(imagens), total_deteccoes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Detecta objetos em imagens locais com modelo/best.pt (sem Roboflow)."
    )
    parser.add_argument(
        "--input", type=Path, default=ROOT / "dataset_amostra/test/images",
        help="Imagem ou pasta de imagens (padrão: dataset_amostra/test/images)"
    )
    parser.add_argument(
        "--model", type=Path, default=ROOT / "modelo/best.pt",
        help="Pesos YOLOv8 (.pt)"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "runs/inferencia_local",
        help="Pasta de saída para imagens anotadas e relatórios"
    )
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confiança mínima, entre 0 e 1")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Dimensão de inferência (padrão: 640)")
    parser.add_argument("--device", default="cpu",
                        help="cpu (padrão) ou índice de GPU CUDA (ex.: 0)")
    args = parser.parse_args(argv)

    try:
        imagens, deteccoes = executar(
            args.input, args.model, args.output, args.conf, args.imgsz, args.device
        )
    except (FileNotFoundError, ValueError, RuntimeError) as erro:
        print(f"Erro: {erro}", file=sys.stderr)
        return 1

    print(f"Concluído: {imagens} imagem(ns), {deteccoes} detecção(ões).")
    print(f"Resultados: {args.output.resolve()}")
    print("Relatórios: deteccoes.csv e resumo.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
