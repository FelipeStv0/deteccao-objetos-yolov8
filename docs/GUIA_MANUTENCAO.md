# Guia de manutenção e integração — YOLOv8

## Onde modificar

| Arquivo | Responsabilidade | Extensão prevista |
| --- | --- | --- |
| `inferir.py` | CLI e orquestração de imagens com YOLOv8 | Novos parâmetros ou pós-processamento de imagem |
| `entrada_imagens.py` | Verificar tipos de entrada e selecionar imagens | Suporte a novos formatos; manter política sem recursão |
| `inferir_video.py` | Captura de vídeo, inferência quadro a quadro e MP4 anotado | Novos codecs, controles de FPS ou stream, com testes |
| `validacao.py` | Validações comuns para imagem e vídeo | Regras de entrada compartilhadas |
| `relatorios.py` | Extrair boxes e produzir `deteccoes.csv`/`resumo.json` | Novas colunas e contrato versionado |
| `avaliar_modelo.py` | Avaliação exploratória da amostra anotada | Novas métricas, preservando protocolo de teste |
| `auditar_dataset.py` | Checar rótulos e integridade da amostra | Auditorias adicionais |

## Contratos de compatibilidade

- `python inferir.py` e `python inferir_video.py` continuam pontos de entrada.
- `inferir.listar_imagens` permanece importável para quem já usava essa função.
- CSV, JSON e nomes dos parâmetros CLI devem permanecer estáveis; novas colunas precisam de testes e documentação.
- Importar os módulos puros não deve carregar PyTorch, OpenCV ou o checkpoint.
- `modelo/best.pt` é um checkpoint de inferência; não representa o dataset integral nem prova generalização.

## Antes de integrar

1. Execute `python -m unittest discover -s tests -v` em um ambiente limpo.
2. Execute uma inferência de demonstração com o checkpoint real e compare CSV/JSON com o comportamento anterior.
3. Faça avaliação separada com rótulos verdadeiros: contagem de detecções não equivale a Precision/Recall/mAP.
4. Não publique imagens/datasets sem verificar autorização de uso e procedência.
5. Registre no PR se os testes usaram simulações ou o modelo real; não extrapole resultados da amostra de dez imagens.
