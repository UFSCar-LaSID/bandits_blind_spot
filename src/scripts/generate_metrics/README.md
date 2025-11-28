<div align="center">
    
Language

English | [Portuguese (BR)](./README_PT-BR.md)
    
</div>

# Generate Metrics

This section is responsible for generating metrics to evaluate the algorithms. It will produce tables and graphs showing metrics across different algorithms and datasets.

## Running the Metric Generation Code

Use the `main.py` script to calculate metrics and save graphics and tables. To run it, execute the following command from the project's root directory:

```
python src/scripts/generate_metrics/main.py
```

Running it this way will prompt for the following inputs:

1.  **Datasets** (you can choose more than one)
2.  **Metrics** (you can choose more than one)
3.  **Non-incremental algorithms** (you can choose more than one, or none at all)
4.  **Incremental algorithms** (you can choose more than one, or none at all)
5.  **Embedding generator for incremental algorithms** (you can choose more than one)
6.  **Context builder for incremental algorithms** (you can choose more than one)

Another way to select options is by running the command below:

```
python src/scripts/generate_metrics/main.py --datasets <datasets> --metrics <metrics> --not_incremental_algorithms <not_incremental_algorithms> --incremental_algorithms <incremental_algorithms> --embeddings <embeddings> --contexts <contexts>
```

Replace `<datasets>` with the names (or indices) of the datasets, separated by commas (","). The available datasets are:

  * \[1\]: amazon-beauty
  * \[2\]: amazon-books
  * \[3\]: amazon-games
  * \[4\]: bestbuy
  * \[5\]: delicious2k
  * \[6\]: delicious2k-urlPrincipal
  * \[7\]: ml-100k
  * \[8\]: ml-25m
  * \[9\]: retailrocket
  * all (will use all datasets)

Replace `<metrics>` with the names (or indices) of the metrics, separated by commas (","). The available metrics are:

  * \[1\]: ncdg
  * \[2\]: hit rate (hr)
  * \[3\]: f-score
  * \[4\]: novelty
  * \[5\]: coverage
  * \[6\]: diversity
  * all (will use all metrics)

Replace `<not_incremental_algorithms>` with the names (or indices) of the non-incremental algorithms, separated by commas (","). The non-incremental algorithms selected here will be used to compare results with the incremental algorithms. The available non-incremental algorithms are:

  * \[1\]: als
  * \[2\]: bpr
  * all (will use all algorithms)

Replace `<incremental_algorithms>` with the names (or indices) of the incremental algorithms, separated by commas (","). The available incremental algorithms to run are:

  * \[1\]: Lin
  * \[2\]: LinUCB
  * \[3\]: LinGreedy
  * \[4\]: LinTS
  * all (will use all algorithms)

Replace `<embeddings>` with the names (or indices) of the embeddings, separated by commas (","). The non-incremental algorithms selected here will be used to find results for incremental algorithms that used the selected embedding options. The available embeddings are:

  * \[1\]: als
  * \[2\]: bpr
  * all (will use all embeddings)

Replace `<contexts>` with the names (or indices) of the context generation strategies, separated by commas (","). The available strategies are:

  * \[1\]: user
  * \[2\]: item\_concat
  * \[3\]: item\_mean
  * \[4\]: item\_concat+user
  * \[5\]: item\_mean+user
  * \[6\]: item\_concat+item\_mean
  * \[7\]: item\_concat+item\_mean+user
  * all (will use all strategies)

-----

The names of the non-incremental algorithms will be the same as those registered in the options table. They will be suffixed with `upper` if they were trained using previous training windows. Otherwise, if they only used the training partition, they will have the `lower` postfix (final name: `<not-incremental-algo-name> - <upper or lower>`).

The names of the incremental algorithms that appear will depend on the chosen embedding generation options and context builder. The names of the incremental algorithms follow this format: `<algo-incremental-name> - <embeddings?> - <context?>` (if only one embedding generator or context builder is chosen, their names will be removed from the incremental algorithm's final name).

At the end of the code execution, several results will be saved, which will be explained in the next section.

### Saved Graphs, Tables, and Other Information

The saved data follows this pattern:

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

Each file/directory is described below:

#### <save_path>

This is the folder where all experiment information will be saved.

It follows the pattern:

`<src.DIR_RESULTS>/<dataset_name>/<current_timestamp>`

#### metadata.json

This is a JSON file following the format below:

```json
{
  "metrics_names": [],              // Names of the metrics used
  "not_incremental_algo_names": [], // Names of the non-incremental algorithms used
  "incremental_algos": {            // Information about incremental algorithms
    "algo_names": [],               // Names of the evaluated incremental algorithms
    "embeddings_names": [],         // Name of the embeddings (which embedding generator was used)
    "contexts_builders_names": []   // Names of the context builders used
  },
  "configs": {
    "<constant>": "<value>"         // Constant values used, such as top-N size, split sizes, etc.
  }
}
```

#### logs.txt

Logs/prints saved by the [`Logger`](/src/scripts/utils/Logger.py) during code execution.

#### window_mean_metrics.csv

This is a table containing the average values obtained across all windows. Rows represent algorithms, and columns represent the different selected metrics.

#### full_test_metrics.csv

This is a table containing the metric calculated for the entire test dataset (it doesn't average across windows, which might slightly change the result). Rows represent algorithms, and columns represent the different selected metrics.

#### per_window_graphic.html, per_window_graphic.png, and per_window_table.csv

For each window, the metric is calculated for all possible algorithms. The graph will show lines representing the algorithms' performance across the windows.

An HTML and PNG version of the graph are generated. The table used to generate the graph is also saved in CSV format.

**Note:** One such graph is generated for each selected metric type.

#### window_agg_graphic.html, window_agg_graphic.png, and window_agg_table.csv

Each window's result is the average result of all preceding windows up to the current window. For example, the result for window 3 is the average result of windows 1, 2, and 3.

An HTML and PNG version of the graph are generated. The table used to generate the graph is also saved in CSV format.

**Note:** One such graph is generated for each selected metric type.
