
# Dataset

Será uma classe utilizada para coletar os dados de interações para todos os experimentos. Deve fazer as seguintes etapas:

- Ler a base de dados (`.csv`)  e guardar em um DataFrame pandas
- Fazer as partições da base de dados: **treino, validação e janelas de teste**
- Remover cold start da partição de teste (itens e usuários não vistos no treino são removidos)
- Limpeza de interações recorrentes (mesmo item consumido, tempo diferente)
- Encode dos ids utilizando `pd.factorize`
- Fornecer funções para posterior decode de ids (recebe lista de ids encoded e transforma para os ids originais)


A base de dados será dividida da seguinte forma:

- **Partição completa de treino**: o tamanho do treino e definido na constante [TRAIN_SIZE](https://github.com/RecSys-UFSCar/mab-recsys/blob/main/src/__init__.py#L47-L48) (por exemplo, se for 0.5, então metade de todas as interações serão usadas para treino)
    - Partição de treino: usada como treino para calcular métricas de desempenho na partição de validação (para buscar a melhor combinação de hiperparâmetros). O tamanho deste treino é 1 - [VAL_SIZE](https://github.com/RecSys-UFSCar/mab-recsys/blob/main/src/__init__.py#L50)
    - Partição de validação: usada para calcular métricas para avaliar o desempenho de diferentes combinações de hiperparâmetros. O tamanho da validação está definido na constante [VAL_SIZE](https://github.com/RecSys-UFSCar/mab-recsys/blob/main/src/__init__.py#L50)
- **Partição completa de teste**: usado para avaliar o desempenho final dos algoritmos. Vale ressaltar que essa partição é filtrada para remover *cold start.* Ou seja, todas interações com usuários e itens que só aparecem na partição de teste (não aparecem no treino) são removidas. O tamanho da partição de teste é 1 - [TRAIN_SIZE](https://github.com/RecSys-UFSCar/mab-recsys/blob/main/src/__init__.py#L47-L48)
    - Janelas de teste: a partição de teste é dividida em N partes iguais (podendo ter 1 interação a mais ou a menos). Útil para avaliar o desempenho incremental dos algoritmos. A quantidade de janelas está definida na constante [TEST_WINDOWS_QNT](https://github.com/RecSys-UFSCar/mab-recsys/blob/main/src/__init__.py#L49)

Essas divisões podem ser verificadas na imagem abaixo:

![dataset_splits_image](https://github.com/RecSys-UFSCar/mab-recsys/blob/Esqueletos%26docs/images/dataset-splits.png)

## Adicionando uma nova base de dados

Todas as bases de dados disponíveis podem ser consultadas na tabela disponível em [`src/datasets/__init__.py`](https://github.com/RecSys-UFSCar/mab-recsys/blob/main/src/datasets/__init__.py)

Para cada base de dados, é necessário que esta possua um arquivo contendo suas interações. Essas interações devem ser salvas em [DIR_DATASET](https://github.com/RecSys-UFSCar/mab-recsys/blob/main/src/__init__.py#L4)/dataset_name/[FILE_INTERACTIONS](https://github.com/RecSys-UFSCar/mab-recsys/blob/main/src/__init__.py#L13).

As interações são dispostas em uma tabela, em um arquivo `.csv`, que deve possuir as seguintes colunas

- [COLUMN_USER_ID](https://github.com/RecSys-UFSCar/mab-recsys/blob/main/src/__init__.py#L19): ID que identifica unicamente um usuário
- [COLUMN_ITEM_ID](https://github.com/RecSys-UFSCar/mab-recsys/blob/main/src/__init__.py#L18): ID que identifica unicamente um item
- [COLUMN_DATETIME](https://github.com/RecSys-UFSCar/mab-recsys/blob/main/src/__init__.py#L21) ou [COLUMN_TIMESTAMP](https://github.com/RecSys-UFSCar/mab-recsys/blob/main/src/__init__.py#L22): momento em que a interação foi realizada
- [COLUMN_RATING](https://github.com/RecSys-UFSCar/mab-recsys/blob/main/src/__init__.py#L20) (opcional): avaliação feita pelo usuário para aquele item