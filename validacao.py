"""Validação compartilhada de entradas para inferência com YOLOv8.

INTEGRAÇÃO FUTURA: centralize aqui verificações usadas tanto em imagens
quanto em vídeos. Não carregue o modelo aqui: a importação de Ultralytics
deve continuar tardia para permitir testes sem dependências pesadas.
"""
from __future__ import annotations

from pathlib import Path


def validar_pesos(pesos: Path) -> None:
    """Impede inferência sem um checkpoint local existente."""
    if not pesos.is_file():
        raise FileNotFoundError(f"Pesos não encontrados: {pesos}")


def validar_parametros(confianca: float, tamanho: int) -> None:
    """Valida limites comuns sem alterar a semântica atual das CLIs."""
    if not 0 <= confianca <= 1:
        raise ValueError("--conf deve estar entre 0 e 1")
    if tamanho <= 0:
        raise ValueError("--imgsz deve ser positivo")
