"""Inferência em vídeo local, preservando resolução e gerando CSV/JSON."""

import argparse
import math
import sys
from pathlib import Path

from relatorios import extrair_deteccoes, salvar_relatorios

ROOT = Path(__file__).resolve().parent
EXTENSOES_VIDEO = {".mp4", ".avi", ".mov", ".mkv"}


def executar_video(entrada: Path, pesos: Path, saida: Path, confianca: float,
                  tamanho: int, dispositivo: str, max_quadros: int = 0) -> tuple[int, int]:
    if not entrada.is_file():
        raise FileNotFoundError(f"Vídeo não encontrado: {entrada}")
    if entrada.suffix.lower() not in EXTENSOES_VIDEO:
        raise ValueError("Use um vídeo MP4, AVI, MOV ou MKV")
    if not pesos.is_file():
        raise FileNotFoundError(f"Pesos não encontrados: {pesos}")
    if saida.suffix.lower() != ".mp4":
        raise ValueError("A saída deve ter extensão .mp4")
    if entrada.resolve() == saida.resolve():
        raise ValueError("O vídeo de saída não pode substituir o arquivo original")
    if not 0 <= confianca <= 1:
        raise ValueError("--conf deve estar entre 0 e 1")
    if tamanho <= 0:
        raise ValueError("--imgsz deve ser positivo")
    if max_quadros < 0:
        raise ValueError("--max-frames não pode ser negativo")

    import cv2
    from ultralytics import YOLO

    captura = cv2.VideoCapture(str(entrada))
    if not captura.isOpened():
        captura.release()
        raise ValueError(f"Não foi possível abrir o vídeo: {entrada}")

    fps = float(captura.get(cv2.CAP_PROP_FPS))
    largura = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not math.isfinite(fps) or fps <= 0 or largura <= 0 or altura <= 0:
        captura.release()
        raise ValueError("Vídeo sem FPS ou dimensões válidas")

    saida = saida.resolve()
    saida.parent.mkdir(parents=True, exist_ok=True)
    gravador = cv2.VideoWriter(
        str(saida), cv2.VideoWriter_fourcc(*"mp4v"), fps, (largura, altura)
    )
    if not gravador.isOpened():
        gravador.release()
        captura.release()
        raise RuntimeError(f"Não foi possível gravar o MP4 em: {saida}")

    registros = []
    quadros = 0
    modelo = None
    try:
        modelo = YOLO(str(pesos))
        while max_quadros == 0 or quadros < max_quadros:
            ok, quadro = captura.read()
            if not ok:
                break
            resultado = modelo.predict(
                source=quadro, conf=confianca, imgsz=tamanho,
                device=dispositivo, verbose=False
            )[0]
            gravador.write(resultado.plot())  # Ultralytics retorna BGR para OpenCV.
            quadros += 1
            registros.extend(
                extrair_deteccoes(
                    resultado, entrada.name, quadro=quadros,
                    tempo_s=round((quadros - 1) / fps, 3)
                )
            )
            if quadros % 25 == 0:
                print(f"Processados: {quadros} quadros")
    finally:
        captura.release()
        gravador.release()

    if quadros == 0:
        saida.unlink(missing_ok=True)
        raise ValueError("O vídeo não contém quadros decodificáveis")

    resumo = {
        "tipo": "video",
        "arquivo_entrada": entrada.name,
        "arquivo_saida": saida.name,
        "checkpoint": pesos.name,
        "fps": fps,
        "largura": largura,
        "altura": altura,
        "quadros_processados": quadros,
        "total_deteccoes": len(registros),
        "parametros": {
            "conf": confianca, "imgsz": tamanho, "device": dispositivo,
            "max_frames": max_quadros,
        },
    }
    salvar_relatorios(saida.parent, registros, resumo)
    return quadros, len(registros)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Detecta objetos em vídeo local com best.pt (sem Roboflow)."
    )
    parser.add_argument("--input", type=Path, default=ROOT / "video/video_original.mp4",
                        help="Caminho para vídeo MP4, AVI, MOV ou MKV")
    parser.add_argument("--model", type=Path, default=ROOT / "modelo/best.pt")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "runs/video_local/video_anotado.mp4",
                        help="Caminho do vídeo MP4 anotado")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Limite opcional de quadros; 0 processa todo o vídeo")
    args = parser.parse_args(argv)

    try:
        quadros, deteccoes = executar_video(
            args.input, args.model, args.output, args.conf, args.imgsz,
            args.device, args.max_frames
        )
    except (FileNotFoundError, ValueError, RuntimeError) as erro:
        print(f"Erro: {erro}", file=sys.stderr)
        return 1

    print(f"Concluído: {quadros} quadros, {deteccoes} detecção(ões).")
    print(f"Vídeo anotado: {args.output.resolve()}")
    print(f"Relatórios: {args.output.resolve().parent / 'deteccoes.csv'} e resumo.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
