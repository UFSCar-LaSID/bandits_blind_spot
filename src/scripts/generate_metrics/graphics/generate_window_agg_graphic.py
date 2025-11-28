
import os

import plotly.graph_objects as go

import src
from src.scripts.generate_metrics.metrics.calculate_all_metrics import MetricsDict
from src.scripts.utils.save import save_dataframe



def generate_window_agg_graphic(save_path: str, calcultated_metrics: MetricsDict):
    '''
    Gera e salva os gráficos das médias agregadas  (para todas as métricas, tanto html, png e csv).

    Cada janela é o resultado médio das janelas anteriores até a janela atual. Por exemplo, o resultado da janela 3 é o resultado médio das janelas 1, 2 e 3. OBS: Um gráfico desse é gerado para cada tipo de métrica selecionado.

    Os seguintes arquivos serão salvos (baseado em `save_path` e `calcultated_metrics`):

    ```
    <save_path>
    ├── graphics
    │   └── <metric>
    │       ├── window_agg_graphic.html
    │       └── window_agg_graphic.png
    └── graphics_tables
        └── <metric>
            └── window_agg_table.csv
    ```

    params:

        calcultated_metrics: Um dicionário seguindo a seguinte estrutura:

        ```
        {
            "metrics": {
                "<metric_name>": {
                    "per_window": "pd.DataFrame",
                    "window_agg": "pd.DataFrame"
                }
            },
            "window_mean": "pd.DataFrame",
            "test_full": "pd.DataFrame"
        } 
        ```

        Os DataFrames demarcados são descritos a seguir:
        
        - `per_window`: é calculado a métrica para cada janela de teste (um DataFrame para cada métrica). As colunas são as janelas, as linhas os algoritmos

        - `window_agg`: cada janela é o resultado médio das janelas anteriores até a janela atual. Por exemplo, o resultado da janela 3 é o resultado médio das janelas 1, 2 e 3. Um DataFrame é criado para cada métrica. As colunas são as janelas, as linhas os algoritmos

        - `window_mean`: é uma tabela que possui a média dos valores obtidos em todas as janelas. As linhas são os algoritmos e as colunas são as diferentes métricas (apenas um DataFrame para todas as métricas)

        - `test_full`: são calculadas as métricas para todo o conjunto de teste (todas as janelas de testes "concatenadas"). As colunas são as métricas, as linhas os algoritmos (apenas um DataFrame para todas as métricas)
    '''
    for metric_name, metric_dict in calcultated_metrics['metrics'].items():
        algos_configs = {}

        window_agg_df = metric_dict['window_agg']
        algo_names = window_agg_df['algo_name'].unique()

        for i, algo_name in enumerate(algo_names):
            algos_configs[algo_name] = {
                'color': src.GRAPHIC_COLORS[i % len(src.GRAPHIC_COLORS)],
                'dash': 'solid'
            }

        fig = go.Figure()
        for algo_name, config in algos_configs.items():
            df_algo = window_agg_df[window_agg_df['algo_name'] == algo_name]
            fig.add_trace(go.Scatter(x=df_algo['window_number'], y=df_algo['value'], mode='lines', name=algo_name, line=dict(color=config['color'], dash=config['dash'])))
        
        fig.update_layout(title=f'{metric_name} x Window number', xaxis_title='Window number', yaxis_title=metric_name)
        # fig.show()

        os.makedirs(os.path.join(save_path, 'graphics', metric_name), exist_ok=True)
        os.makedirs(os.path.join(save_path, 'graphics_tables', metric_name), exist_ok=True)

        fig.write_html(os.path.join(save_path, 'graphics', metric_name, src.FILE_WINDOW_AGG_GRAPHIC_HTML))
        fig.write_image(os.path.join(save_path, 'graphics', metric_name, src.FILE_WINDOW_AGG_GRAPHIC_PNG))

        save_dataframe(
            window_agg_df,
            os.path.join(save_path, 'graphics_tables', metric_name, src.FILE_WINDOW_AGG_TABLE)
        )