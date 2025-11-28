<div align="center">
    
Idioma

Português (BR) | [Inglês](./README.md)
    
</div>

# Experimentos incrementais

Os experimentos incrementais consiste no treinamento e geração de recomendações usando algoritmos que possuem um treinamento incremental (ou seja, estes não precisam retreinar "do zero" para adicionar novos dados no seu conhecimento). Por exemplo, o LinUCB e LinGreedy são considerados incrementais.

## Executando os experimentos

O código `main.py` deve ser usado para executar um conjunto de (ou apenas um) experimento não-incremental. Para executá-lo, utilize o comando abaixo, no repositório raiz do projeto:

```
python src/scripts/incremental/main.py
```

Executando dessa forma, serão requisitados quatro inputs:

1. Algoritmo incremental (e.g. LinUCB, LinGreedy, LinTemporal)
2. Base de dados
3. Algoritmo das embeddings (e.g. ALS e BPR)
4. Função de geração de contextos (e.g. users embeddings, items mean, items concat, etc.)

Outra forma de selecionar as opções é executando o comando abaixo:

```
python src/scripts/incremental/main.py --algorithms <algorithms> --datasets <datasets> --embeddings <embeddings> --contexts <contexts>
```

Substitua `<algorithms>` pelos nomes (ou índices) dos algoritmos incrementais separados por vírgula (","). Os algoritmos disponíveis para execução são:

- \[1\]: Lin
- \[2\]: LinUCB
- \[3\]: LinGreedy
- \[4\]: LinTS
- all (usará todos os algoritmos)

Substitua `<datasets>` pelos nomes (ou índices) dos conjuntos de dados separados por vírgula (","). Os conjuntos de dados disponíveis para uso como treino/teste são:

- \[1\]: amazon-beauty
- \[2\]: amazon-books
- \[3\]: amazon-games
- \[4\]: bestbuy
- \[5\]: delicious2k
- \[6\]: delicious2k-urlPrincipal
- \[7\]: ml-100k
- \[8\]: ml-25m
- \[9\]: retailrocket
- all (usará todos os conjuntos de dados)

Substitua `<embeddings>` pelos nomes (ou índices) dos algoritmos não incrementais separados por vírgula (","). As embeddings geradas por esses algoritmos serão usadas como parte do contexto MAB. Portanto, é necessário gerar as embeddings antes. As embeddings disponíveis são:

- \[1\]: als
- \[2\]: bpr
- all (usará todas as embeddings)

Substitua `<contexts>` pelos nomes (ou índices) das estratégias de geração de contexto separadas por vírgula (","). As estratégias disponíveis são:

- \[1\]: user
- \[2\]: item\_concat
- \[3\]: item\_mean
- \[4\]: item\_concat+user
- \[5\]: item\_mean+user
- \[6\]: item\_concat+item\_mean
- \[7\]: item\_concat+item\_mean+user
- all (usará todas as estratégias)

Se mais de uma opção for selecionada em um item, todas as possíveis combinações serão executadas.

## O que é feito em cada experimento ?

Antes mesmo de iniciar o experimento, a base de dados é carregada e todo pré-processamento e splits são feitos. Para mais informações sobre esta etapa, [consulte a documentação a respeito das bases de dados](/src/scripts/preprocess/README_PT-BR.md).

Com a base de dados carregada, todas combinações de hiperparâmetros são avaliadas. No caso, todas as combinações possíveis entre hiperparâmetros do algoritmo incremental e das embeddings serão testadas. Para essa avaliação, o algoritmo é treinado na partição de **TREINO** e depois avaliado na partição de **VALIDAÇÃO**, calculando-se uma métrica (como NDCG ou HR). Para cada combinação de hiperparamêtros são salvas as recomendações na partição de **VALIDAÇÃO**.

Com a melhor combinação de hiperparâmetros (que obteve melhor NDCG), o algoritmo é treinado e testado de forma incremental (semelhante ao que é feito no **upper bound** dos algoritmos não-incrementais). Primeiro é treinado com a base de treino completa e recomendações para a primeira janela de teste são feitas. Depois, é treinado utilizando a base de treino completa + primeira janela de teste e recomendações para a segunda janela de teste. Segue dessa forma até realizar recomendações para todas as janelas. Feito isso, recomendações para todas as janelas de teste serão salvas.

Esta avaliação é exemplificada logo abaixo:

![incremental-protocol-example](/images/incremental_eval.png)

Ao fim do experimento, vários resultados serão salvos, os quais vão ser explicados na próxima seção.

### Dados salvos por um experimento

Os dados salvos seguem o seguinte padrão:

```
<save_path>
├── metadata.json
├── logs.txt
├── final_recommendations.csv
└── <hiperparam_combination_id>
    └── recommendations.csv
```

Cada arquivo / repositório é descrito a seguir:

#### <save_path>

É a pasta onde todas informações do experimento serão salvas.

Segue o padrão abaixo:

`<experiments-dir>/<dataset_name>/incremental/<incremental_algo_name>/<embeddings_algo_name>/<build_context_name>/<timestamp_atual>`

#### metadata.json

```json
{
    "grid_search": {
		"<hiperparam_combination_id>": {               // Definido mais a baixo. È um número inteiro que identifica cada combinação de hiperparâmetro 
			"params": {                                // Objeto contendo os hiperparâmetros testados
				"incremental_params": {                // Hiperparams. testados especificos do algo. incrmental
					"<param_name>": "<param_value>"
				},
				"embeddings_params": {                 // Hiperparams. testados especificos das embeddings
					"<param_name>": "<param_value>"
				},
				"contexts_params": {                   // Hiperparams. testados especificos de funções criadoras de contexto
					"<param_name>": "<param_value>"
				}
			},
			"score": "<value>"                         // Pontuação obtida (e.g. NDCG, HR) utilizando esses parâmetros no conjunto de validação
		}
	},
	"best_hiperparam_combination_id": "ID",            // Id da combinação de hiperparâmetros que obteve melhor score
	"configs": {
		"<constant>": "<value>"                        // Valores constantes utilizados, como tamanho top-N, splits size, etc.
	}
}
```

#### logs.txt

Logs/prints salvos pelo [`Logger`](/src/scripts/utils/Logger.py) durante a execução de um experimento.

As informações aqui podem variar muito de experimento para experimento, podendo salvar informações de tempo demorado para fazer alguma etapa no código, métricas calculadas, etc.

#### final_recommendations.csv

É um arquivo `.csv` que vai armazenar as recomendações feitas para TODAS as partições de teste por um algoritmo incremental (com a MELHOR combinação de hiperparâmetros). Será armazenado uma lista de recomendações e scores PARA CADA INTERAÇÃO (até mesmo as interações negativas).

Esta tabela possui as seguintes colunas:

1. **COLUMN_USER_ID**: id do usuário que fez a interação. O id deve ser o id original contido na base de dados.
2. **COLUMN_ITEM_ID**: id do item que foi consumido. O id deve ser o id original contido na base de dados.
3. **COLUMN_RATING**: avaliação / pontuação armazenada na base de dados para essa interação
5. **COLUMN_WINDOW_NUMBER**: número da partição de teste que essa interação faz parte (de 0 a número de janelas - 1)
6. **COLUMN_DATETIME**: momento em que ocorreu a interação, fornecido pela própria base de dados.
7. **COLUMN_RECOMMENDATIONS**: uma lista top-N contendo os IDs dos itens recomendados. Deve estar no formato de uma lista do Python (ex: `[1, 2, 3]`)
8. **COLUMN_SCORES**: uma lista do mesmo tamanho da lista de recomendações, mapeando cada item recomendado à um score (gerado pelo modelo de aprendizado). Deve estar no formato de uma lista do Python (ex: `[0.5, 0.4, 0.3]`)

#### <hiperparam_combination_id>

Um valor identificando unicamente uma combinação de hiperparâmetros. O id escolhido para representar as combinações foram números inteiros positivos sequenciais (0, 1, 2, etc)

#### recommendations.csv

É um arquivo `.csv` que vai armazenar as recomendações feitas na partição de validação por um algoritmo incremental (em uma combinação de hiperparâmetros). Será armazenado uma lista de recomendações e scores PARA CADA INTERAÇÃO (até mesmo as interações negativas).

Esta tabela possui as seguintes colunas:

1. **COLUMN_USER_ID**: id do usuário que fez a interação. O id deve ser o id original contido na base de dados.
2. **COLUMN_ITEM_ID**: id do item que foi consumido. O id deve ser o id original contido na base de dados.
3. **COLUMN_RATING**: avaliação / pontuação armazenada na base de dados para essa interação
4. **COLUMN_DATETIME**: momento em que ocorreu a interação, fornecido pela própria base de dados.
5. **COLUMN_RECOMMENDATIONS**: uma lista top-N contendo os IDs dos itens recomendados. Deve estar no formato de uma lista do Python (ex: `[1, 2, 3]`)
6. **COLUMN_SCORES**: uma lista do mesmo tamanho da lista de recomendações, mapeando cada item recomendado à um score (gerado pelo modelo de aprendizado). Deve estar no formato de uma lista do Python (ex: `[0.5, 0.4, 0.3]`)
