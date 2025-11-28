<div align="center">
    
Idioma

Português (BR) | [Inglês](./README.md)
    
</div>

# Experimentos não-incrementais

Os experimentos não-incrementais consiste no treinamento e geração de recomendações usando algoritmos que não possuem um treinamento incremental (ou seja, estes precisam retreinar "do zero" para adicionar novos dados no seu conhecimento). Por exemplo, o ALS e o BPR são considerados não-incrementais.

## Executando os experimentos

O código `main.py` deve ser usado para executar um conjunto de (ou apenas um) experimento não-incremental. Para executá-lo, utilize o comando abaixo, no repositório raiz do projeto:

```
python src/scripts/not_incremental/main.py
```

Executando dessa forma, serão requisitados dois inputs:

1. Bases de dados a serem exploradas. Devem ser identificadas por ID's, separados por espaço (verifique as bases de dados disponíveis em [`src/datasets/__init__.py`](/src/datasets/__init__.py)
2. Algoritmos (não-incrementais) a serem utilizados. Devem ser identificados por ID's, separados por espaço (verifique os algoritmos disponíveis em [`src/algorithms/not_incremental/__init__.py`](/src/algorithms/not_incremental/__init__.py)

Outra forma de selecionar as opções é executando o comando abaixo:

```
python src/scripts/not_incremental/main.py --algorithms <algorithms> --datasets <datasets>
```

Substitua `<algorithms>` pelos nomes (ou índices) dos algoritmos separados por vírgula (","). Os algoritmos disponíveis para execução são:

- \[1\]: als
- \[2\]: bpr
- all (usará todos os algoritmos)

Substitua `<datasets>` pelos nomes (ou índices) das bases de dados separadas por vírgula (","). Os conjuntos de dados disponíveis para uso como treino/teste são:

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

Com os conjuntos de algoritmos e bases de dados a serem executados, todas as combinações possíveis de experimentos serão realizadas. Ou seja, cada algoritmo será testado em cada uma das bases de dados selecionadas.

## O que é feito em cada experimento ?

Antes mesmo de iniciar o experimento, a base de dados é carregada e todo pré-processamento e splits são feitos. Para mais informações sobre esta etapa, consulte a [documentação a respeito do pré-processamento](/src/scripts/preprocess/README_PT-BR.md).
Com a base de dados carregada, todas combinações de hiperparâmetros são avaliadas. Os hiperparâmetros a serem testados dos algoritmos não-incrementais podem ser encontrados no [arquivo de configuração de hiperparâmetros](/src/algorithms/not_incremental/hyperparameters.py).
Para essa avaliação, o algoritmo é treinado na partição de **treino** e depois avaliado na partição de **validação**, calculando-se uma métrica (como NDCG ou HR).
Para cada combinação de hiperparâmetros são salvas as embeddings de usuários e itens aprendidas utilizando a partição de **treino**. Também são salvas as recomendações feitas na partição de **validação**. Com a melhor combinação de hiperparâmetros (que obteve o melhor valor na métrica avaliada), são geradas as embeddings utilizando o **treino completo** e dois resultados:

- **lower_bound:** o algoritmo é treinado na base de **treino completa** e são feitas as recomendações em **toda a base de teste**. Essa avaliação está exemplificada na imagem abaixo.

  ![lower-bound-example](/images/lower_bound.png)

- **upper_bound**:  o algoritmo é treinado na base de **treino completa** e depois vai treinando, interativamente, nas **janelas de teste**. Ou seja, primeiro é treinado com a base de treino completa e recomendações para a primeira janela de teste são feitas. Depois, é treinado utilizando a base de treino completa + primeira janela de teste e recomendações para a segunda janela de teste. Segue dessa forma até realizar recomendações para todas as janelas. Essa avaliação está exemplificada na imagem abaixo.

  ![upper-bound-example](/images/upper_bound.png)

Ao fim do experimento, vários resultados serão salvos, os quais vão ser explicados na próxima seção.

### Dados salvos por um experimento

Os dados salvos seguem o seguinte padrão:

```
<save_path>
├── metadata.json
├── logs.txt
├── recommendations_lower.csv
├── recommendations_upper.csv
└── <hiperparam_combination_id>
    ├── train_embeddings
    │   ├── users_embeddings.npy
    │   └── items_embeddings.npy
    ├── train_full_embeddings
    │   ├── users_embeddings.npy
    │   └── items_embeddings.npy
    └── recommendations.csv
```

Cada arquivo / repositório é descrito a seguir:

#### <save_path>

É a pasta onde todas informações do experimento serão salvas.

Segue o padrão abaixo:

`experiments/<dataset_name>/not_incremental/<algo_name>/<timestamp_atual>`

#### metadata.json

É um arquivo JSON seguindo o formato abaixo:

```json
{
  "grid_search": {
    "<hiperparam_combination_id>": {      // Definido mais a baixo. È um número inteiro que identifica cada combinação de hiperparâmetro unicamente
      "params": {                         // Objeto contendo os hiperparâmetros testados
        "<param_name>": "<param_value>"
      },
      "score": "<value>"                  // Pontuação obtida (e.g. NDCG, HR) utilizando esses parâmetros no conjunto de validação
    }
  },
  "best_hiperparam_combination_id": "ID",  // Id da combinação de hiperparâmetros que obteve melhor score
  "configs": {
    "<constant>": "<value>"               // Valores constantes utilizados, como tamanho top-N, splits size, etc.
  }
}
```

#### logs.txt

Logs/prints salvos pelo [`Logger`](/src/scripts/utils/Logger.py) durante a execução de um experimento.

As informações aqui podem variar muito de experimento para experimento, podendo salvar informações de tempo demorado para fazer alguma etapa no código, métricas calculadas, etc.

#### recommendations_lower.csv e recommendations_upper.csv

É um arquivo `.csv` que vai armazenar as recomendações feitas para TODAS as partições de teste por um algoritmo não incremental (com a MELHOR combinação de hiperparâmetros). Será armazenado uma lista de recomendações e scores PARA CADA INTERAÇÃO (até mesmo as interações negativas).

No caso, em `recommendations_lower.csv`, algoritmo será treinado apenas com a **partição de treino completa**.

Já para o `recommendations_upper.csv`, o algoritmo será treinado com a **partição de treino completa + as janelas de teste anteriores** (por exemplo, para as recomendações da janela de número 3, o algoritmo será treinado com o treino + janelas 0, 1 e 2 de teste)

Esta tabela possui as seguintes colunas:

1. **COLUMN_USER_ID**:  id do usuário que fez a interação. O id deve ser o id original contido na base de dados.
2. **COLUMN_ITEM_ID**: id do item que foi consumido. O id deve ser o id original contido na base de dados.
3. **COLUMN_RATING**: avaliação / pontuação armazenada na base de dados para essa interação
4. **COLUMN_WINDOW_NUMBER**: número da partição de teste que essa interação faz parte (de 0 a número de janelas - 1)
5. **COLUMN_DATETIME**: momento em que ocorreu a interação, fornecido pela própria base de dados.
6. **COLUMN_RECOMMENDATIONS**: uma lista top-N contendo os IDs dos itens recomendados. Deve estar no formato de uma lista do Python (ex: `[1, 2, 3]`)
7. **COLUMN_SCORES**: uma lista do mesmo tamanho da lista de recomendações, mapeando cada item recomendado à um score (gerado pelo modelo de aprendizado). Deve estar no formato de uma lista do Python (ex: `[0.5, 0.4, 0.3]`)

#### <hiperparam_combination_id>

Um valor identificando unicamente uma combinação de hiperparâmetros. O id escolhido para representar as combinações foram números inteiros positivos sequenciais (0, 1, 2, etc)

#### train_embeddings

Pasta contendo as embeddings geradas a partir do treinamento utilizando a **partição de treino** (apenas o treino, não inclui a validação)

As embeddings nesse caso são geradas para **todas as combinações de hiperparâmetros**.

#### train_full_embeddings

Pasta contendo as embeddings geradas a partir do treinamento utilizando a **partição de treino completa (treino + validação)**.

Só será gerado para a **melhor combinação de hiperparâmetros** obtidas no experimento não-incremental. Mais dessas embeddings podem ser geradas nos experimentos incrementais, sob demanda, caso as melhores embeddings utilizadas na busca em grade incremental não possua a sua versão full.

#### recommendations.csv

É um arquivo `.csv` que vai armazenar as recomendações feitas na partição de validação por um algoritmo não incremental (em uma combinação de hiperparâmetros). Será armazenado uma lista de recomendações e scores PARA CADA INTERAÇÃO (até mesmo as interações negativas).

Esta tabela possui as seguintes colunas:

1. **COLUMN_USER_ID**: id do usuário que fez a interação. O id deve ser o id original contido na base de dados.
2. **COLUMN_ITEM_ID**: id do item que foi consumido. O id deve ser o id original contido na base de dados.
3. **COLUMN_RATING**: avaliação / pontuação armazenada na base de dados para essa interação
4. **COLUMN_DATETIME**: momento em que ocorreu a interação, fornecido pela própria base de dados.
5. **COLUMN_RECOMMENDATIONS**: uma lista top-N contendo os IDs dos itens recomendados. Deve estar no formato de uma lista do Python (ex: `[1, 2, 3]`)
6. **COLUMN_SCORES**: uma lista do mesmo tamanho da lista de recomendações, mapeando cada item recomendado à um score (gerado pelo modelo de aprendizado). Deve estar no formato de uma lista do Python (ex: `[0.5, 0.4, 0.3]`)

#### users_embeddings.npy

São as embeddings de usuários. Estão salvas no padrão do Numpy, podendo ser lidas através do comando [`np.load`](https://numpy.org/doc/stable/reference/generated/numpy.load.html).

Ao ler usando esse comando, uma matriz numpy será retornada com as embeddings dos usuários. Cada linha da matriz é uma embedding para um usuário específico. A primeira linha (índice 0), por exemplo, é a embedding do primeiro usuário que apareceu na base de dados de interações, o índice 1 é o segundo usuário, etc.

OBS: As embeddings não utilizam os ids originais presentes nas bases de dados. Os ids foram todos transformados utilizando o comando `pd.factorize`.

#### items_embeddings.npy

São as embeddings de itens. Estão salvas no padrão do Numpy, podendo ser lidas através do comando [`np.load`](https://numpy.org/doc/stable/reference/generated/numpy.load.html).

Ao ler usando esse comando, uma matriz numpy será retornada com as embeddings dos itens. Cada linha da matriz é uma embedding para um item específico. A primeira linha (índice 0), por exemplo, é a embedding do primeiro item que apareceu na base de dados de interações, o índice 1 é o segundo item, etc.

OBS: As embeddings não utilizam os ids originais presentes nas bases de dados. Os ids foram todos transformados utilizando o comando `pd.factorize`.
