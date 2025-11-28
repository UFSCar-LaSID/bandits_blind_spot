<div align="center">
    
Idioma

Português (BR) | [Inglês](./README.md)
    
</div>

# Pré-processamento das bases de dados

É necessário baixar e pré-processar as bases de dados para que seja possível executar posteriores experimentos.

O código de pré-processamento é responsável por remover todas as interações repetidas das bases de dados, sendo aquelas que possuem item, usuário e momento de interação iguais. No caso, apenas a primeira das interações de um conjunto de interações repetidas é mantida, restando nenhuma duplicata nos conjuntos de dados. Também, são deletadas todas as interações com dados faltantes. Os nomes das colunas das tabelas de interações são padronizados para que todas as bases de dados possam ser carregadas posteriormente da mesma forma (com um mesmo código). Finalmente, é gerado o campo de timestamp baseado na coluna datetime caso essa informação não exista, e vice-versa.

## Bases de dados

Instale as bibliotecas nos locais indicados para que seja possível realizar o préprocessamento da mesma. A seguir é fornecido uma lista de bases de dados que podem ser baixadas e pré-processadas:

- [AmazonBeauty](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/All_Beauty.jsonl.gz): coloque o arquivo `All_Beauty.jsonl` em `raw/amazon-beauty`
- [AmazonBooks](https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews): faça o download e extraia em `raw/amazon-books`
- [AmazonGames](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Video_Games.jsonl.gz): coloque o arquivo `Video_Games.jsonl` em `raw/amazon-games`
- [BestBuy](https://www.kaggle.com/c/acm-sf-chapter-hackathon-big/data?select=train.csv): coloque o arquivo `train.csv` em `raw/BestBuy`
- [Delicious2K](https://grouplens.org/datasets/hetrec-2011): faça o download de `hetrec2011-delicious-2k.zip` na seção `Delicious Bookmarks` e extraia-o em `raw/delicious2k`
- [Delicious2k-urlPrincipal](https://grouplens.org/datasets/hetrec-2011): o mesmo que `Delicious2K`
- [MovieLens-100K](https://grouplens.org/datasets/movielens/): faça o download de `ml-100k.zip` na seção `MovieLens 100K Dataset` e extraia-o em `raw/ml-100k`
- [MovieLens-25M](https://grouplens.org/datasets/movielens/): faça o download de `ml-25m.zip` na seção `MovieLens 25M Dataset` e extraia-o em `raw/ml-25m`
- [RetailRocket](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset): coloque o arquivo `events.csv` em `raw/RetailRocket`

## Executando o código de pré-processamento

Com as bases de dados instaladas, é necessário executar o código de pré-processamento para utilizá-las para o treinamento e geração de recomendações
Para tal, execute o comando a seguir:

```
python src/scripts/preprocess/main.py
```

Executando desta forma, será solicitado que digite as bases de dados a serem pré-processadas. Digite os índices das bases de dados que deseja pré-processar separados por espaço

Outra forma de escolher as bases de dados a serem processadas é fornecendo o argumento `--datasets` na execução do algoritmo. 

Para tal, o comando a ser utilizado é:

```
python src/scripts/preprocess.py --datasets <datasets>
```

Troque `<datasets>` pelos nomes (ou índices) separados por espaçp. As bases disponíveis para pré-processamento são as seguintes:

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
