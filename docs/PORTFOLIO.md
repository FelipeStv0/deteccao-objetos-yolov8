# Caso técnico para portfólio — detecção de objetos com YOLOv8

## Problema

Construir um detector para três categorias de acessórios — bolsas,
óculos e relógios — capaz de inferir sobre imagens e vídeos locais sem
depender do serviço utilizado na coleta original de dados.

## Solução e decisões técnicas

1. **Separação de responsabilidades:** o notebook mantém a trilha do
   experimento original; os scripts CLI de imagem/vídeo usam os pesos
   salvos sem Roboflow.
2. **Saídas auditáveis:** cada predição pode ser inspecionada em CSV/JSON,
   incluindo classes, scores, coordenadas e arquivo/quadro de origem.
3. **Qualidade de código:** testes unitários de parsing, IO, caminhos,
   geração de relatórios e pareamento de caixas; testes de integração
   em CPU com checkpoint real.
4. **Avaliação exploratória:** auditoria de labels, mAP do Ultralytics
   no pequeno split de teste versionado, identificação de falsos
   positivos/negativos, e benchmark de latência com warmup.
5. **Proveniência:** resultados guardam configurações, versões e hash
   do checkpoint. Os logs e artefatos do GitHub Actions são referências
   verificáveis, não números inventados.

## Demonstração

Comece pela [execução local](REPRODUCAO.md), execute a inferência nas
10 imagens de teste e inspecione as caixas anotadas. Em seguida,
execute a [avaliação exploratória](REPRODUCAO.md#3-avaliação-exploratória-e-análise-de-erros)
e analise o CSV de erros para discutir falhas do detector.

### Exemplo de narrativa para entrevista

> O projeto original treinou um detector YOLOv8 para três classes.
> Para transformá-lo em um projeto reproduzível de engenharia, separei
> a inferência das dependências externas, acrescentei exportações,
> testes e automação. Implementei auditoria dos rótulos e uma avaliação
> exploratória de apenas dez imagens com comparação a bounding boxes
> anotadas. Expliquei explicitamente que essa amostra não permite
> afirmar generalização, e registrei latência e ambiente de execução
> para orientar os próximos experimentos.

Adapte esse texto para refletir **somente as contribuições que você
efetivamente realizou e consegue explicar**.

## Perguntas que o projeto permite discutir

- Por que IoU e NMS são relevantes na detecção de objetos?
- Qual a diferença entre score de confiança, Precision, Recall e mAP?
- Por que medir mAP em dez imagens não valida generalização?
- O que pode dar errado com rótulos, divisões e vazamento de dados?
- Por que o benchmark precisa de warmup e descrição de hardware?
- O que unit tests verificam que um teste real com best.pt não verifica?
- Como projetar uma avaliação independente antes de colocar o modelo
  em produção?

## Próxima etapa científica

Obter, com autorização, o conjunto integral e um teste externo
independente, revisar anotações, medir desempenho por classe e
documentar erros de domínio. Até lá, a evidência é de **engenharia
reproduzível e experimentação limitada**, não de um detector validado
para produção.
