# Detecção de objetos com YOLOv8

Projeto de visão computacional com treinamento supervisionado de um detector YOLOv8 para identificar **bolsas, óculos e relógios** em imagens e vídeos.

## Objetivo e abordagem

O estudo utiliza uma base de aproximadamente 500 imagens, com anotações no formato YOLO e divisões de treino, validação e teste. O modelo base informado é o `yolov8s.pt`, treinado por 50 épocas com imagens de 640 × 640 pixels em Google Colab (GPU Tesla T4).

**Tecnologias:** Python, Ultralytics YOLOv8, PyTorch, OpenCV, Pillow, Matplotlib e Pandas.

## Resultados relatados

O relatório original apresenta, para a validação identificada como **ONNX, etapa 8**, os seguintes resultados gerais:

| Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| ---: | ---: | ---: | ---: |
| 0,806 | 0,770 | 0,818 | 0,535 |

Essas métricas são **resultados documentados**, ainda não reproduzidos independentemente nesta preparação. Entre as classes, relógios tiveram maior precisão (0,871) e óculos maior recall (0,848). O desempenho é limitado pelo número de imagens, condições de captura e casos de oclusão.

## Arquivos do repositório

- `README.md`: apresentação e resultados documentados.
- `requirements.txt`: dependências para o ambiente Python.

**A publicação do notebook, do relatório técnico, dos gráficos e dos pesos treinados ainda está pendente.** Por isso, este repositório ainda não fornece uma reprodução completa.

## Execução (após publicação do notebook)

O treinamento usa um dataset preparado no padrão YOLO. Ao publicar o notebook, os caminhos e as instruções de obtenção dos dados precisam estar consistentes. Credenciais do Roboflow devem ser fornecidas por variável de ambiente (por exemplo, `ROBOFLOW_API_KEY`), **nunca no repositório**.

O dataset e os vídeos não devem ser redistribuídos sem conferir as licenças e permissões correspondentes.

## Melhorias futuras

Ampliar e diversificar a base, avaliar aumento de dados (*data augmentation*) e comparar variantes maiores do YOLOv8. Essas propostas não são funcionalidades já implementadas.
