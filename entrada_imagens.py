"""Seleção de entradas de imagem; sem dependência de PyTorch ou YOLO.

INTEGRAÇÃO FUTURA: altere EXTENSOES para novos formatos e acrescente testes.
A coleta é superficial por contrato: não busca imagens em subpastas.
"""
from __future__ import annotations
from pathlib import Path

EXTENSOES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def listar_imagens(entrada: Path) -> list[Path]:
    """Seleciona uma imagem ou todas as imagens diretas de um diretório."""
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
