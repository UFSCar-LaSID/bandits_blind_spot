<div align="center">
    
Idioma

Português (BR) | [Inglês](./README.md)
    
</div>

# Geração de métricas

Responsável por gerar as métricas para avaliação dos algoritmos. Serão geradas tabelas e gráficos com métricas entre diferentes algoritmos em diferentes bases de dados.

## Executando o código de geração de métricas

O código `main.py` deve ser usado para calcular as métricas e salvar gráficos e tabelas. Para executá-lo, utilize o comando abaixo, no repositório raiz do projeto:

```
python src/scripts/generate_metrics/main.py
```

Executando dessa forma, os seguintes inputs serão solicitados:

1. Base de dados (pode escolher mais de uma)
2. Métricas (pode escolher mais de uma)
3. Escolher os algoritmos não-incremental (pode escolher mais de um, e até mesmo não escolher nenhum)
4. Escolher os algoritmos incrementais (pode escolher mais de um, e até mesmo não escolher nenhum)
5. Escolher o gerador de embeddings dos algoritmos incrementais (pode escolher mais de um)
6. Escolher o construtor de contextos dos algoritmos incrementais (pode escolher mais de um)

Outra forma de selecionar as opções é executando o comando abaixo:

```
python src/scripts/generate_metrics/main.py --datasets <datasets> --metrics <metrics> --not_incremental_algorithms <not_incremental_algorithms> --incremental_algorithms <incremental_algorithms> --embeddings <embeddings> --contexts <contexts>
```

Substitua `<datasets>` pelos nomes (ou índices) dos conjuntos de dados separados por vírgula (","). Os conjuntos de dados disponíveis para uso são:

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

Substitua `<metrics>` pelos nomes (ou índices) das métricas separadas por vírgula (","). As métricas disponíveis são:

- \[1\]: ncdg
- \[2\]: hit rate (hr)
- \[3\]: f-score
- \[4\]: novelty
- \[5\]: coverage
- \[6\]: diversity
- all (usará todas as métricas)

Substitua `<not_incremental_algorithms>` pelos nomes (ou índices) dos algoritmos não incrementais separados por vírgula (","). Os algoritmos não incrementais selecionados aqui serão usados para comparar os resultados com os algoritmos incrementais. Os algoritmos não incrementais disponíveis são:

- \[1\]: als
- \[2\]: bpr
- all (usará todos os algoritmos)

Substitua `<incremental_algorithms>` pelos nomes (ou índices) dos algoritmos incrementais separados por vírgula (","). Os algoritmos incrementais disponíveis para execução são:

- \[1\]: Lin
- \[2\]: LinUCB
- \[3\]: LinGreedy
- \[4\]: LinTS
- all (usará todos os algoritmos)

Substitua `<embeddings>` pelos nomes (ou índices) das embeddings separados por vírgula (","). Os algoritmos não incrementais selecionados aqui serão usados para encontrar resultados sobre algoritmos incrementais que usaram as embeddings das opções de embedding selecionadas. As embeddings disponíveis são:

- \[1\]: als
- \[2\]: bpr
- all (usará todos as embeddings)

Substitua `<contexts>` pelos nomes (ou índices) das estratégias de geração de contexto separadas por vírgula (","). As estratégias disponíveis são:

- \[1\]: user
- \[2\]: item\_concat
- \[3\]: item\_mean
- \[4\]: item\_concat+user
- \[5\]: item\_mean+user
- \[6\]: item\_concat+item\_mean
- \[7\]: item\_concat+item\_mean+user
- all (usará todas as estratégias)

Os nomes dos algoritmos não-incrementais serão os mesmos que estão cadastrados na tabela de opções. Estes serão acrescentados de `upper` se estes foram treinados usando as janelas anteriores de treino. Caso contrário, se usaram apenas a partição de treino, terá o pós-fixo de `lower` (nome final: `<not-incremental-algo-name> - <upper or lower>`)

O nome dos algoritmos incrementais que apareceram dependerão das opções de geração de embeddings escolhidas e de construtor de contexto. O nome dos algoritmos incrementais segue o seguinte formato: `<algo-incremental-name> - <embeddings?> - <context?>` (se for escolhido apenas um gerador de embeddings ou construtor de contextos, o nome destes serão removidos do nome final do algoritmo incremental).

Ao fim da execução do código, vários resultados serão salvos, os quais serão explicados na próxima seção.

### Gráficos, tabelas e outras informações salvas

Os dados salvos seguem o seguinte padrão:

```
<save_path>
├── metadata.json
├── logs.txt
├── window_mean_metrics.csv
├── full_test_metrics.csv
├── graphics
│   └── <metric>
│       ├── per_window_graphic.html
│       ├── per_window_graphic.png
│       ├── window_agg_graphic.html
│       └── window_agg_graphic.png
└── graphics_tables
    └── <metric>
        ├── per_window_table.csv
        └── window_agg_table.csv
```

Cada arquivo / repositório é descrito a seguir:

#### <save_path>

É a pasta onde todas informações do experimento serão salvas.

Segue o padrão abaixo:

`<src.DIR_RESULTS>/<dataset_name>/<timestamp_atual>`

#### metadata.json

É um arquivo JSON seguindo o formato abaixo:

```json
{
  "metrics_names": [],                  // Nomes das métricas utilizadas
  "not_incremental_algo_names": [],     // Nomes dos algoritmos não incrementais usados
  "incremental_algos": {                // Informações dos algoritmos incrementais
    "algo_names": [],                   // Nomes dos algoritmos incrementais avaliados
    "embeddings_names": [],             // Nome das embeddings (qual foi o gerador de embeddings) utilizados
    "contexts_builders_names": []       // Nome dos construtores de contexto utilizados
  },
  "configs": {
    "<constant>": "<value>"             // Valores constantes utilizados, como tamanho top-N, splits size, etc.
  }
}
```

#### logs.txt

Logs/prints salvos pelo [`Logger`](/src/scripts/utils/Logger.py) durante a execução do código.

#### window_mean_metrics.csv

É uma tabela que possui a média dos valores obtidos em todas as janelas. As linhas são os algoritmos e as colunas são as diferentes métricas escolhidas.

#### full_test_metrics.csv

É uma tabela que possui a métrica calculada para a base completa de teste (não faz a média das janelas, isso pode mudar um pouco o resultado). As linhas são os algoritmos e as colunas são as diferentes métricas escolhidas.

#### per_window_graphic.html, per_window_graphic.png e per_window_table.csv

Para cada janela é calculado a métrica para todos os algoritmos possíveis. O gráfico terá linhas com o desempenho dos algoritmos ao longo das janelas.

É gerado uma versão HTML e png do gráfico. Também é salvo a tabela que foi usada para gerar o gráfico no formato CSV.

OBS: Um gráfico desse é gerado para cada tipo de métrica selecionado.

#### window_agg_graphic.html, window_agg_graphic.png e window_agg_table.csv

Cada janela é o resultado médio das janelas anteriores até a janela atual. Por exemplo, o resultado da janela 3 é o resultado médio das janelas 1, 2 e 3. 

É gerado uma versão HTML e png do gráfico. Também é salvo a tabela que foi usada para gerar o gráfico no formato CSV.

OBS: Um gráfico desse é gerado para cada tipo de métrica selecionado.
