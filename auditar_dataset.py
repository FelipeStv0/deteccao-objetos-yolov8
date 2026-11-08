"""Audita a amostra YOLO: pares imagem/rótulo, caixas e duplicações exatas.

Não substitui avaliação de qualidade, revisão humana ou auditoria do dataset integral.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPLITS = ("train", "valid", "test")
EXTENSOES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CLASSES = ("bolsa", "oculos", "relogio")


def _hash_arquivo(arquivo: Path) -> str:
    digest = hashlib.sha256()
    with arquivo.open("rb") as f:
        for bloco in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(bloco)
    return digest.hexdigest()


def validar_linha(rotulo: str, origem: str, linha: int,
                  n_classes: int = len(CLASSES)) -> int:
    partes = rotulo.split()
    if len(partes) != 5:
        raise ValueError(f"{origem}:{linha}: esperado formato 'classe cx cy w h'")
    try:
        classe = int(partes[0])
        v = [float(x) for x in partes[1:]]
    except ValueError as erro:
        raise ValueError(f"{origem}:{linha}: rótulo inválido") from erro
    if not 0 <= classe < n_classes:
        raise ValueError(f"{origem}:{linha}: ID de classe fora do intervalo: {classe}")
    if not all(0 <= x <= 1 for x in v):
        raise ValueError(f"{origem}:{linha}: coordenadas devem estar entre 0 e 1")
    cx, cy, w, h = v
    if not 0 < w <= 1 or not 0 < h <= 1:
        raise ValueError(f"{origem}:{linha}: largura/altura inválidas")
    tolerancia = 0.00001
    if (cx - w / 2 < -tolerancia or cx + w / 2 > 1 + tolerancia
            or cy - h / 2 < -tolerancia or cy + h / 2 > 1 + tolerancia):
        raise ValueError(f"{origem}:{linha}: caixa fora da imagem")
    return classe


def auditar(base: Path) -> dict:
    base = base.resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"Dataset não encontrado: {base}")

    resumo = {"dataset": str(base), "divisoes": {}, "erros": [],
              "duplicados_entre_divisoes": [], "avisos": []}
    hashes = defaultdict(list)
    nomes_originais = defaultdict(list)

    for split in SPLITS:
        imagens_dir = base / split / "images"
        labels_dir = base / split / "labels"
        if not imagens_dir.is_dir() or not labels_dir.is_dir():
            resumo["erros"].append(f"{split}: diretórios images/labels ausentes")
            continue

        imagens = sorted(x for x in imagens_dir.iterdir()
                         if x.is_file() and x.suffix.lower() in EXTENSOES)
        labels = {x.stem: x for x in labels_dir.glob("*.txt") if x.is_file()}
        contagem = Counter()
        sem_rotulos = []
        for imagem in imagens:
            nomes_originais[imagem.stem.split("_jpg.rf.")[0]].append((split, imagem.name))
            hashes[_hash_arquivo(imagem)].append((split, imagem.name))
            label = labels.pop(imagem.stem, None)
            if label is None:
                resumo["erros"].append(f"{split}: rótulo ausente de {imagem.name}")
                continue
            linhas = [x.strip() for x in label.read_text(encoding="utf-8").splitlines()
                      if x.strip()]
            if not linhas:
                sem_rotulos.append(imagem.name)
            for numero, linha in enumerate(linhas, 1):
                try:
                    contagem[validar_linha(linha, str(label), numero)] += 1
                except ValueError as erro:
                    resumo["erros"].append(str(erro))

        for rotulo in labels.values():
            resumo["erros"].append(f"{split}: rótulo sem imagem: {rotulo.name}")
        resumo["divisoes"][split] = {
            "imagens": len(imagens),
            "caixas": sum(contagem.values()),
            "por_classe": {CLASSES[i]: contagem[i] for i in range(len(CLASSES))},
            "imagens_sem_caixas": sem_rotulos,
        }

    for grupo in hashes.values():
        if len({s for s, _ in grupo}) > 1:
            resumo["duplicados_entre_divisoes"].append(grupo)
    if resumo["duplicados_entre_divisoes"]:
        resumo["erros"].append("Imagens com SHA-256 idêntico em divisões diferentes")

    # Os nomes de Roboflow podem esconder transformações; isso é um alerta, não
    # uma prova de vazamento ou de duplicação perceptual.
    for origem, grupo in nomes_originais.items():
        if origem and len({s for s, _ in grupo}) > 1:
            resumo["avisos"].append({
                "tipo": "possivel_mesma_origem_pelo_nome",
                "identificador": origem, "arquivos": grupo,
            })

    resumo["aprovado"] = not resumo["erros"]
    return resumo


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Auditar a amostra de rótulos YOLO.")
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset_amostra")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/auditoria_dataset.json")
    args = parser.parse_args(argv)
    try:
        resultado = auditar(args.dataset)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(resultado, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (ValueError, OSError) as erro:
        print(f"Erro: {erro}", file=sys.stderr)
        return 2
    for split, dados in resultado["divisoes"].items():
        print(f"{split}: {dados['imagens']} imagens, {dados['caixas']} caixas")
    print(f"Problemas: {len(resultado['erros'])}; avisos: {len(resultado['avisos'])}")
    print(f"Relatório: {args.output.resolve()}")
    return 0 if resultado["aprovado"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
