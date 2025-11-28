
import sys
import os

parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(parent_path)

from src.scripts.utils.parameters_handle import get_input
from src.datasets import DATASETS_TABLE
import src


def main():
    '''
    Código responsável por pré-processar um conjunto de bases de dados. É possível fornecer mais de uma base de dados por execução (basta responder com os números das bases de dados separados por espaço).

    O pré-processamento é responsável por remover todas as interações repetidas das bases de dados, sendo aquelas que possuem item, usuário e momento de interação iguais. No caso, apenas a primeira das interações de um conjunto de interações repetidas é mantida, restando nenhuma duplicata nos conjuntos de dados. Também, são deletadas todas as interações com dados faltantes. Os nomes das colunas das tabelas de interações são padronizados para que todas as bases de dados possam ser carregadas posteriormente da mesma forma (com um mesmo código). Finalmente, é gerado o campo de timestamp baseado na coluna datetime caso essa informação não exista, e vice-versa.
    '''
    options = get_input('Choose datasets to preprocess', [
        {
            'name': 'datasets',
            'description': 'Dataset names (or indexes) to preprocess. If not provided, a interactive menu will be shown. If "all" is provided, all datasets will be preprocessed.',
            'options': DATASETS_TABLE,
            'name_column': src.DATASET_NAME
        }
    ])[0]

    for option_index in options:
        dataset_name = DATASETS_TABLE.loc[option_index, src.DATASET_NAME]
        print('Preprocessing {}...'.format(dataset_name))
        preprocess_function = DATASETS_TABLE.loc[option_index, src.DATASET_PREPROCESS_FUNCTION]
        input_path = os.path.join(src.DIR_RAW, dataset_name)
        output_path = os.path.join(src.DIR_DATASET, dataset_name)
        preprocess_function(input_path, output_path)
        print('Preprocessing {} done!'.format(dataset_name))


if __name__ == '__main__':
    main()
    sys.exit(0)