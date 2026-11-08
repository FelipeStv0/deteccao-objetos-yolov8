"""Inferência local de imagens com os pesos YOLOv8 deste repositório."""

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXTENSOES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def listar_imagens(entrada: Path) -> list[Path]:
    """Valida uma imagem ou lista imagens da pasta (sem subpastas)."""
    if not entrada.exists():
        raise FileNotFoundError(f"Entrada não encontrada: {entrada}")
    if entrada.is_file():
        if entrada.suffix.lower() not in EXTENSOES:
            raise ValueError(f"Formato de imagem não suportado: {entrada.suffix}")
        return [entrada]
    if not entrada.is_dir():
        raise ValueError(f"A entrada deve ser um arquivo ou diretório: {entrada}")
    imagens = sorted(
        p for p in entrada.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSOES
    )
    if not imagens:
        raise ValueError(f"Nenhuma imagem compatível encontrada em: {entrada}")
    return imagens


def executar(entrada: Path, pesos: Path, saida: Path, confianca: float,
             tamanho: int, dispositivo: str) -> tuple[int, int]:
    if not pesos.is_file():
        raise FileNotFoundError(f"Pesos não encontrados: {pesos}")
    if not 0 <= confianca <= 1:
        raise ValueError("--conf deve estar entre 0 e 1")
    if tamanho <= 0:
        raise ValueError("--imgsz deve ser positivo")

    imagens = listar_imagens(entrada)
    if saida.resolve() == entrada.resolve() or (
        entrada.is_file() and saida.resolve() == entrada.parent.resolve()
    ):
        raise ValueError("A pasta de saída não pode substituir as imagens originais")

    # Importação tardia: --help e os testes de caminhos não exigem PyTorch.
    from ultralytics import YOLO

    modelo = YOLO(str(pesos))
    total_deteccoes = 0
    for imagem in imagens:
        resultados = modelo.predict(
            source=str(imagem),
            conf=confianca,
            imgsz=tamanho,
            device=dispositivo,
            save=True,
            project=str(saida.resolve().parent),
            name=saida.name,
            exist_ok=True,
            verbose=False,
        )
        quantidade = sum(len(resultado.boxes) for resultado in resultados)
        total_deteccoes += quantidade
        print(f"{imagem.name}: {quantidade} detecção(ões)")

    return len(imagens), total_deteccoes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Detecta objetos em imagens locais com modelo/best.pt (sem Roboflow)."
    )
    parser.add_argument("--input", type=Path,
                        default=ROOT / "dataset_amostra/test/images",
                        help="Imagem ou diretório local (padrão: amostra de teste)")
    parser.add_argument("--model", type=Path, default=ROOT / "modelo/best.pt",
                        help="Checkpoint .pt (padrão: modelo/best.pt)")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/inferencia_local",
                        help="Diretório para imagens anotadas")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confiança mínima, de 0 a 1 (padrão: 0.25)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Tamanho de inferência (padrão: 640)")
    parser.add_argument("--device", default="cpu",
                        help="Dispositivo: cpu ou 0 para a primeira GPU CUDA")
    args = parser.parse_args(argv)

    try:
        imagens, deteccoes = executar(
            args.input, args.model, args.output, args.conf, args.imgsz, args.device
        )
    except (FileNotFoundError, ValueError) as erro:
        print(f"Erro: {erro}", file=sys.stderr)
        return 1

    print(f"Concluído: {imagens} imagem(ns), {deteccoes} detecção(ões).")
    print(f"Resultados anotados: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
