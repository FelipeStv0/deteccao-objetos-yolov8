# Guia de reprodução — Windows e CI

## 1. Pré-requisitos

Python 3.12 ou 3.11, PowerShell e espaço para PyTorch.
A inferência pode ser feita em CPU, sem CUDA, API ou Roboflow.

Na raiz do clone ou ZIP atualizado:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-inferencia.txt
```

Se já criou `.venv` com Python 3.12, reutilize-o.
Para executar CLI sem ativar o ambiente virtual, use sempre
` .\.venv\Scripts\python.exe ` (sem espaços ao digitar).

## 2. Verificar a integridade da amostra

```powershell
.\.venv\Scripts\python.exe auditar_dataset.py
```

A auditoria gera `runs/auditoria_dataset.json` e falha
com código diferente de zero se encontrar rótulos ausentes, órfãos,
coordenadas inválidas ou duplicações **binárias** exatas entre splits.
Outras limitações podem persistir.

## 3. Avaliação exploratória e análise de erros

```powershell
.\.venv\Scripts\python.exe avaliar_modelo.py --device cpu --warmup 3 --repeats 15
explorer .\runs\avaliacao_amostra
```

Saídas:
- `auditoria_dataset.json` — validações de estrutura;
- `dataset_avaliado.yaml` — configuração com **caminho absoluto**;
- `metricas_amostra.json` — métricas Ultralytics sobre **10 imagens**;
- `erros.csv` — erros de pareamento por imagem, IoU=0,5;
- `benchmark.json` — latências de imagem única, após warmup;
- `validacao_ultralytics/` — arquivos auxiliares de validação.

**Importante:** trata-se de uma avaliação de uma amostra pequena, 
não de verificação do relatório original nem estimativa robusta de 
generalização. O YAML publicado em `dataset_amostra/data.yaml`
descreve classes e estrutura; o script gera **seu próprio YAML absoluto**
para evitar erro ao executar fora da raiz do repositório.

## 4. Testar imagens e vídeos

```powershell
.\.venv\Scripts\python.exe inferir.py --input "dataset_amostra\test\images"
.\.venv\Scripts\python.exe inferir_video.py --max-frames 30
```

Para um teste real com sua própria pasta:

```powershell
.\.venv\Scripts\python.exe inferir.py --input "$env:USERPROFILE\Downloads\teste_yolo" --output "runs\imagens_proprias"
```

Arquivos em Downloads permanecem locais e não são enviados ao GitHub.

## 5. Testes automatizados

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

O CI também gera artefatos para auditoria, avaliação e benchmark.
A seção Summary da execução de avaliação lista os resultados **da
execução específica**. Para comparar latências entre máquinas,
padronize CPU/GPU, número de repetições, aquecimento, imagem,
versão do checkpoint e software. Cada benchmark inclui leitura da
imagem e pré/pós-processamento, mas exclui carregar o modelo e
gravar imagens no disco.

## 6. Reprodução do treino integral (não executada neste fluxo)

É necessário acesso autorizado à mesma versão do dataset completo,
rótulos e `data.yaml`, além da configuração do treino
registrada no notebook. Não confundir `dataset_amostra` com
o conjunto original. Consulte [model card](MODEL_CARD.md).

Os resultados da avaliação amostral jamais devem substituir as métricas
do relatório original; publique ambos separadamente, com protocolo, data,
código e referência ao job do GitHub Actions.
