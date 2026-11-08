# Model Card — YOLOv8: bolsas, óculos e relógios

**Estado:** demonstração acadêmica / portfólio, não produto de produção.
**Arquitetura:** YOLOv8s, implementado com Ultralytics/PyTorch.
**Checkpoint distribuído:** [modelo/best.pt](../modelo/best.pt).
**Treinamento relatado:** 50 épocas, imagens 640×640, Colab (Tesla T4).
**Classes identificadas na amostra:** 0 bolsa; 1 óculos; 2 relógio.
**Data da revisão desta ficha:** 2026-11-08.

## Finalidade e escopo

O projeto demonstra treinamento supervisionado previamente realizado,
inferência local em imagens/vídeos, avaliação exploratória com rótulos
e infraestrutura de testes. Adequado para aprendizado e demonstrações
técnicas. Não é indicado para decisões de segurança, vigilância, 
aplicações médicas, identificação de indivíduos ou automação crítica.

## Dados e proveniência

O experimento original relata cerca de 500 imagens e usa um dataset
obtido via workspace Roboflow. O conjunto integral, seu versionamento,
licenças/termos completos e procedimento completo de aquisição não estão
embutidos neste repositório. Consulte também [link_dataset.txt](../link_dataset.txt).

O diretório [dataset_amostra](../dataset_amostra) contém **somente 30 imagens**
distribuídas em 10 de treino, 10 de validação e 10 de teste, acompanhadas
das respectivas anotações YOLO. Esses arquivos não constituem uma
republicação do conjunto completo e não permitem afirmar que o teste
representa uma amostra independente da população real de uso.

A auditoria [auditar_dataset.py](../auditar_dataset.py) verifica:
pares imagem/rótulo, integridade das coordenadas normalizadas, classe
válida, arquivos órfãos e arquivos com SHA-256 idêntico em splits
distintos. Ela **não detecta automaticamente duplicação visual/perceptual,
erros humanos de anotação ou vazamento de conteúdo semântico**.

## Resultados e verificações — interpretar separadamente

| Evidência | O que demonstra | O que NÃO demonstra |
| --- | --- | --- |
| Testes unitários | comportamento do código/CLI | qualidade do detector |
| CI de inferência real | carregamento de best.pt, JPG e MP4 gerados | precisão das classes |
| Avaliação exploratória do teste reduzido | métricas calculadas para **10 imagens anotadas** | generalização / reprodução das métricas originais |
| Benchmark em CI (CPU) | latência na infraestrutura efêmera de GitHub Actions | FPS em PC específico ou capacidade de produção |

Os resultados históricos do relatório PDF são **relatados**, não
verificados independentemente com o dataset integral:

| Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| ---: | ---: | ---: | ---: |
| 0,806 | 0,770 | 0,818 | 0,535 |

A execução de [avaliar_modelo.py](../avaliar_modelo.py) salva
`metricas_amostra.json`, `erros.csv`,
`benchmark.json` e a auditoria. Também salva um YAML
com os nomes de classes lidos diretamente do checkpoint. Os resultados
devem ser rotulados como **exploratórios** para uma amostra pequena.

## Análise de erros

Com confiança = 0,25 e IoU = 0,5, a análise adicional faz pareamento
greedy de predição/rótulo da mesma classe e registra por imagem:
verdadeiros positivos, falsos positivos e falsos negativos.
Essas taxas no ponto de operação **não equivalem** à implementação de
Precision/Recall/mAP com varredura de thresholds do Ultralytics.
Inspecione os casos individualmente; nem toda caixa divergente
representa uma falha de rotulagem ou uma falha do detector.

## Limitações conhecidas

- Tamanho muito pequeno e procedência incompleta da amostra.
- Possível desequilíbrio de classes, condições de luz e oclusão.
- Não foi estabelecida uma amostra externa, independente, com novas
  cenas/usuários para estimar desempenho fora de distribuição.
- Não foram medidas robustez, calibração da confiança, consistência temporal,
  sensibilidade a grupos/contextos ou consumo energético.
- Benchmarks em CI podem variar entre execuções e máquinas.
- Treinamento original completo permanece dependente dos dados e acessos
  necessários; imagens de terceiros exigem revisão de direitos de uso.

## Riscos e boas práticas

Não interpretar score de confiança como probabilidade calibrada;
não usar detecções para estabelecer identidade, comportamento ou
intenção de pessoas; evitar distribuir fotografias particulares sem
consentimento e autorização de direitos autorais. Verificar licenças
da origem dos pesos/dataset antes de redistribuição comercial.

## Reprodutibilidade e manutenção

- [Execução do notebook e treinamento original](../notebook.ipynb)
  (requer acesso autorizado à origem dos dados).
- [Guia de reprodução e comandos](REPRODUCAO.md), incluindo a dependência de avaliação `PyYAML` no arquivo `requirements-avaliacao.txt`.
- [Plano de avaliação rigorosa futura](AVALIACAO.md).
- [GitHub Actions](https://github.com/FelipeStv0/deteccao-objetos-yolov8/actions).

Os artefatos de avaliação registram versão do Python, PyTorch,
Ultralytics, ambiente de execução, hash do checkpoint e dependências
resolvidas. Números publicados precisam apontar para a execução
que os produziu, e não para um valor inserido manualmente.
