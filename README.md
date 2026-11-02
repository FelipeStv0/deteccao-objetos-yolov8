# Detecção de objetos com YOLOv8

Projeto acadêmico de visão computacional para treinamento, validação e inferência com a biblioteca Ultralytics YOLOv8, usando dataset do Roboflow.

## Arquivos
- `notebook.ipynb`: treinamento e avaliação no Google Colab.
- `resultados/`: gráficos e imagens de avaliação.
- `prints_treino/`: registros do treinamento.
- `modelo/best.pt`: pesos treinados; a redistribuição deve respeitar os termos de uso dos dados.

## Execução
1. Abra `notebook.ipynb` no Google Colab, preferencialmente com GPU.
2. Configure a variável de ambiente `ROBOFLOW_API_KEY` com uma chave sua, sem colocá-la no código público.
3. Instale as bibliotecas em `requirements.txt` e execute as células na sequência.
4. Ajuste os caminhos para o dataset, o modelo e os vídeos utilizados.

**Segurança:** uma chave exposta no notebook original foi retirada da cópia preparada. A chave antiga deve ser revogada no Roboflow.

O dataset completo e os vídeos não acompanham este projeto. As métricas não foram reproduzidas nesta preparação.
