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
- `notebook.ipynb`: treinamento, avaliação e demonstrações (com credencial do Roboflow via variável de ambiente).
- `modelo/best.pt`: pesos treinados para inferência local.
- `Relatorio_YOLOv8_Portfolio.pdf`: relatório técnico de resultados.
- `resultados/`: gráficos e imagens de avaliação.

O treinamento completo ainda depende de acesso autorizado ao dataset original; as métricas acima são as documentadas no relatório, não uma reprodução independente.

## Execução

O treinamento usa um dataset preparado no padrão YOLO. O notebook revisado não contém uma chave de API escrita no código; qualquer acesso futuro ao Roboflow exige uma credencial própria e autorizada, fornecida fora do GitHub (por exemplo, por `ROBOFLOW_API_KEY`). Sem acesso ao workspace original, não é possível garantir o download do mesmo dataset para reproduzir o treinamento.

**Inferência sem Roboflow:** o modelo treinado pode ser carregado localmente com a biblioteca Ultralytics para detectar objetos em novas imagens. Isso não exige a chave do Roboflow.

O dataset e os vídeos não devem ser redistribuídos sem conferir as licenças e permissões correspondentes.

## Inferência local com o checkpoint treinado

Esta etapa **não utiliza Roboflow**, API, token, Google Colab ou o dataset de treinamento completo. O repositório já inclui 'modelo/best.pt' e imagens de demonstração em 'dataset_amostra/test/images'. As predições usam exclusivamente arquivos locais.

### Passo a passo (Windows / PowerShell)

Recomendado: Python 3.11 (ou 3.12), Git e ambiente com acesso à internet apenas para instalar os pacotes na primeira vez.

1. Clone e entre na pasta do projeto:

   ```powershell
   git clone https://github.com/FelipeStv0/deteccao-objetos-yolov8.git
   cd deteccao-objetos-yolov8
   ```

2. Crie um ambiente virtual e instale **somente** as dependências necessárias para inferência:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements-inferencia.txt
   ```

   Se o Python 3.11 não estiver instalado, use `py -3.12` (caso possua o 3.12) ou instale uma das versões. Não é necessário ativar o ambiente virtual.

3. Execute a demonstração sobre as imagens já incluídas:

   ```powershell
   .\.venv\Scripts\python.exe inferir.py
   ```

   As imagens anotadas serão salvas em `runs/inferencia_local/`. Para uma imagem própria, passe seu caminho local entre aspas:

   ```powershell
   .\.venv\Scripts\python.exe inferir.py --input "C:\Imagens\minha_foto.jpg"
   ```

   Para uma pasta de imagens e outro diretório de resultados:

   ```powershell
   .\.venv\Scripts\python.exe inferir.py --input "C:\Imagens" --output "runs\minhas_predicoes" --conf 0.30
   ```

O script varre apenas as imagens **diretamente dentro** da pasta indicada (sem subpastas). Formatos: JPG, JPEG, PNG, BMP, WEBP e TIFF. Ao terminar, informa a quantidade de imagens processadas e detecções e salva imagens com caixas e rótulos. Não modifica os arquivos de entrada. Por padrão usa CPU, sem exigir CUDA. Se houver GPU NVIDIA com PyTorch/CUDA corretamente configurado, é possível usar `--device 0`. Para ver as opções: `python inferir.py --help`.

### Testes e limites de verificação

- Testes automatizados de lógica/CLI: `python -m unittest discover -s tests -v`. Foram executados localmente: **11 testes passaram**. Eles usam um **modelo simulado**; não medem a acurácia nem validam a execução dos pesos reais.
- Workflow do GitHub Actions: executa esses testes em Linux/Windows e tenta uma inferência **real**, em CPU, com `modelo/best.pt` e uma imagem versionada. A execução só está comprovada quando o job `inferencia-real` terminar com sucesso na aba Actions.
- A execução end-to-end com o checkpoint real **não foi verificada neste ambiente**, pois aqui não foi possível instalar os pacotes/obter os binários necessários pela rede. Nenhum resultado de detecção real é inventado neste README.
- **Inferência não é avaliação**: observar caixas sobre imagens não reproduz Precision, Recall, mAP@0.5 ou mAP@0.5:0.95. Essas métricas continuam sendo apenas as relatadas no documento original, **não independentemente verificadas**.
- O treinamento integral e a reavaliação rigorosa exigem o dataset original, suas divisões e as anotações correspondentes. A amostra reduzida versionada não substitui essa base.

Para verificar as predições reais no seu computador, execute o passo 3 e inspecione os arquivos gerados. Se aparecer `ModuleNotFoundError: ultralytics`, confirme que instalou as dependências no mesmo ambiente usado para executar o script. Se aparecer `Pesos não encontrados`, verifique a presença de `modelo/best.pt` ou informe o caminho com `--model`.

## Melhorias futuras

Ampliar e diversificar a base, avaliar aumento de dados (*data augmentation*) e comparar variantes maiores do YOLOv8. Essas propostas não são funcionalidades já implementadas.
