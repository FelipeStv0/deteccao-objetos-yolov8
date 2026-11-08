# Avaliação, evidências e limitações

## Verificações distintas

1. **Unidade:** modelo simulado; valida entrada, parâmetros, lógica e formato CSV/JSON.
2. **Integração:** checkpoint `modelo/best.pt` real em CPU; verifica geração de imagens e vídeo e consistência dos arquivos de saída.
3. **Avaliação supervisionada:** exige anotações de referência independentes e um conjunto de teste fixo para calcular Precision, Recall, mAP@0.5 e mAP@0.5:0.95. **Não foi reproduzida nesta atualização.**

### Evidências anteriores

- [Execução #42500205606](https://github.com/FelipeStv0/deteccao-objetos-yolov8/actions/runs/42500205606) confirmou **1 imagem**, **1 detecção** e arquivo anotado de **95.080 bytes** usando `best.pt`.
- O responsável relatou execução Windows/Python 3.12 com **10 imagens versionadas e 12 detecções**, além do processamento bem-sucedido de **10 fotos adicionais**. Os resultados particulares não foram fornecidos como artefatos nem anotados para avaliação independente.
- A nova execução do GitHub Actions compara o número de registros do CSV aos totais do JSON e verifica os arquivos produzidos. Isso **não** verifica a correção semântica das caixas.

## Estrutura dos dados exportados

Cada linha de `deteccoes.csv` descreve uma **predição**, não um objeto confirmado:

| Campo | Significado |
| --- | --- |
| `arquivo` | Nome do arquivo de entrada |
| `classe_id`, `classe` | ID e rótulo do checkpoint |
| `confianca` | Score do detector entre 0 e 1 |
| `x1, y1, x2, y2` | Extremos da bounding box em pixels |
| `quadro`, `tempo_s` | Número e instante do quadro; vazios para fotos |

O JSON traz `resumo` e `deteccoes`. Imagens sem caixas são contabilizadas no resumo, e CSV sem detecções tem somente cabeçalho. Não derive Precision, Recall ou mAP apenas desses valores.

## Reprodução futura de métricas

A análise adequada depende de recuperar **com autorização** o dataset original, o arquivo `data.yaml`, classes, anotações, divisões e condições equivalentes de teste. Por exemplo, depois de providenciar essa base:

```python
from ultralytics import YOLO

modelo = YOLO("modelo/best.pt")
resultado = modelo.val(
    data="CAMINHO_AUTORIZADO/data.yaml",
    split="test",
    imgsz=640,
    device="cpu",
)
print("mAP@0.5:", resultado.box.map50)
print("mAP@0.5:0.95:", resultado.box.map)
```

**Esse código é um roteiro de avaliação, não um teste já realizado.** Documente versão de dataset, separação, classes e ambiente antes de comparar números. O conjunto pequeno versionado não substitui o dataset integral.

## Limites não resolvidos

- Falsos positivos/negativos e classes trocadas;
- mudanças de domínio, iluminação, oclusão e resolução;
- precisão por classe e testes com rotulagem humana;
- métricas de desempenho em CPU e GPU específicas;
- licença e qualidade do conjunto de treinamento completo.

Métricas que constam no relatório técnico do projeto são **relatadas**, não revalidadas independentemente por esta execução.
