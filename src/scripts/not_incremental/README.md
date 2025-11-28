<div align="center">
    
Language

English | [Portuguese (BR)](./README_PT-BR.md)
    
</div>

# Non-Incremental Experiments

Non-incremental experiments involve training and generating recommendations using algorithms that don't have incremental training (meaning they need to retrain "from scratch" to incorporate new data). For example, ALS and BPR are considered non-incremental.

## Running the Experiments

The `main.py` code should be used to run a set of (or just one) non-incremental experiment. To execute it, use the command below from the project's root directory:

```
python src/scripts/not_incremental/main.py
```

Running it this way will prompt for two inputs:

1.  **Databases to be explored.** These should be identified by indexes, separated by spaces (check available datasets in [`src/datasets/__init__.py`](/src/datasets/__init__.py)).
2.  **Non-incremental algorithms to be used.** These should be identified by indexes, separated by spaces (check available algorithms in [`src/algorithms/not_incremental/__init__.py`](/src/algorithms/not_incremental/__init__.py)).

-----

Another way to select options is by running the command below:

```
python src/scripts/not_incremental/main.py --algorithms <algorithms> --datasets <datasets>
```

Replace `<algorithms>` with the names (or indices) of the algorithms, separated by commas (","). The available algorithms to run are:

  * \[1\]: als
  * \[2\]: bpr
  * all (will use all algorithms)

Replace `<datasets>` with the names (or indices) of the datasets, separated by commas (","). The datasets available for training/testing are:

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

Once you've specified the algorithms and datasets, all possible combinations of experiments will be run. This means each algorithm will be tested on every selected dataset.

## What Happens in Each Experiment?

Even before the experiment begins, the database is loaded, and all preprocessing and splits are performed. For more information on this step, consult the [preprocessing documentation](/src/scripts/preprocess/README_PT-BR.md).

With the database loaded, all hyperparameter combinations are evaluated. The hyperparameters to be tested for the non-incremental algorithms can be found in the [hyperparameter configuration file](https://www.google.com/search?q=/src/algorithms/not_incremental/hyperparameters.py).
For this evaluation, the algorithm is trained on the **training partition** and then evaluated on the **validation partition**, calculating the metric NDCG.

For each hyperparameter combination, the learned user and item embeddings from the **training partition** are saved. The recommendations made on the **validation partition** are also saved. With the best hyperparameter combination (which achieved the best value on the evaluated metric), embeddings are generated using the **full training data**, and two results are produced:

 * **lower_bound:** The algorithm is trained on the **full training data** and recommendations are made across the **entire test dataset**. This evaluation is illustrated in the image below.

   ![lower-bound-example](/images/lower_bound_eng.png)

 * **upper\_bound:** The algorithm is trained on the **full training data** and then incrementally trained on the **test windows**. That is, it's first trained with the full training data, and recommendations for the first test window are made. Then, it's trained using the full training data + the first test window, and recommendations for the second test window are made. This continues until recommendations are made for all windows. This evaluation is illustrated in the image below.

   ![upper-bound-example](/images/upper_bound_eng.png)

At the end of the experiment, several results will be saved, which will be explained in the next section.

### Data Saved by an Experiment

The saved data follows this pattern:

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

Each file/directory is described below:

#### <save_path>

This is the folder where all experiment information will be saved.

It follows the pattern:

`experiments/<dataset_name>/not_incremental/<algo_name>/<current_timestamp>`

#### metadata.json

This is a JSON file following the format below:

```json
{
  "grid_search": {
    "<hiperparam_combination_id>": {      // Defined below. This is an integer that uniquely identifies each hyperparameter combination.
      "params": {                         // Object containing the tested hyperparameters
        "<param_name>": "<param_value>"
      },
      "score": "<value>"                  // Score obtained (e.g., NDCG, HR) using these parameters on the validation set.
    }
  },
  "best_hiperparam_combination_id": "ID", // ID of the hyperparameter combination that achieved the best score.
  "configs": {
    "<constant>": "<value>"               // Constant values used, such as top-N size, split sizes, etc.
  }
}
```

#### logs.txt

Logs/prints saved by the [`Logger`](/src/scripts/utils/Logger.py) during an experiment's execution.

The information here can vary greatly from experiment to experiment, potentially including time taken for certain code steps, calculated metrics, etc.

#### recommendations_lower.csv and recommendations_upper.csv

These are `.csv` files that will store the recommendations made for ALL test partitions by a non-incremental algorithm (with the **BEST hyperparameter combination**). A list of recommendations and scores for **EACH INTERACTION** (even negative interactions) will be stored.

In `recommendations_lower.csv`, the algorithm will be trained only with the **full training partition**.

For `recommendations_upper.csv`, the algorithm will be trained with the **full training partition + previous test windows** (for example, for recommendations for window number 3, the algorithm will be trained with the training data + test windows 0, 1, and 2).

This table has the following columns:

1.  **COLUMN_USER_ID**: ID of the user who made the interaction. The ID should be the original ID contained in the database.
2.  **COLUMN_ITEM_ID**: ID of the item that was consumed. The ID should be the original ID contained in the database.
3.  **COLUMN_RATING**: Rating / score stored in the database for this interaction.
4.  **COLUMN_WINDOW_NUMBER**: Number of the test partition this interaction is part of (from 0 to number of windows - 1).
5.  **COLUMN_DATETIME**: Time at which the interaction occurred, provided by the database itself.
6.  **COLUMN_RECOMMENDATIONS**: A top-N list containing the IDs of recommended items. It should be in the format of a Python list (e.g., `[1, 2, 3]`).
7.  **COLUMN_SCORES**: A list of the same size as the recommendations list, mapping each recommended item to a score (generated by the learning model). It should be in the format of a Python list (e.g., `[0.5, 0.4, 0.3]`).

#### <hiperparam_combination_id>

A value uniquely identifying a hyperparameter combination. The IDs chosen to represent the combinations are sequential positive integers (0, 1, 2, etc.).

#### train_embeddings

A folder containing embeddings generated from training using the **training partition** (only training, not including validation).

Embeddings in this case are generated for **all hyperparameter combinations**.

#### train_full_embeddings

A folder containing embeddings generated from training using the **full training partition (training + validation)**.

This will only be generated for the **best hyperparameter combination** obtained in the non-incremental experiment. More of these embeddings can be generated in incremental experiments, on demand, if the best embeddings used in the incremental grid search do not have their full version.

#### recommendations.csv

This is a `.csv` file that will store recommendations made on the validation partition by a non-incremental algorithm (for a specific hyperparameter combination). It will store a list of recommendations and scores for **EACH INTERACTION** (even negative interactions).

This table has the following columns:

1.  **COLUMN_USER_ID**: ID of the user who made the interaction. The ID should be the original ID contained in the database.
2.  **COLUMN_ITEM_ID**: ID of the item that was consumed. The ID should be the original ID contained in the database.
3.  **COLUMN_RATING**: Rating / score stored in the database for this interaction.
4.  **COLUMN_DATETIME**: Time at which the interaction occurred, provided by the database itself.
5.  **COLUMN_RECOMMENDATIONS**: A top-N list containing the IDs of recommended items. It should be in the format of a Python list (e.g., `[1, 2, 3]`).
6.  **COLUMN_SCORES**: A list of the same size as the recommendations list, mapping each recommended item to a score (generated by the learning model). It should be in the format of a Python list (e.g., `[0.5, 0.4, 0.3]`).

#### users_embeddings.npy

These are user embeddings. They are saved in Numpy format and can be read using the [`np.load`](https://numpy.org/doc/stable/reference/generated/numpy.load.html) command.

When read using this command, a Numpy matrix will be returned with the user embeddings. Each row of the matrix is an embedding for a specific user. The first row (index 0), for example, is the embedding of the first user that appeared in the interaction database, index 1 is the second user, and so on.

**Note:** The embeddings do not use the original IDs present in the databases. All IDs have been transformed using the `pd.factorize` command.

#### items_embeddings.npy

These are items embeddings. They are saved in Numpy format and can be read using the [`np.load`](https://numpy.org/doc/stable/reference/generated/numpy.load.html) command.

When read using this command, a Numpy matrix will be returned with the item embeddings. Each row of the matrix is an embedding for a specific item. The first row (index 0), for example, is the embedding of the first item that appeared in the interaction database, index 1 is the second item, and so on.

**Note:** The embeddings do not use the original IDs present in the databases. All IDs have been transformed using the `pd.factorize` command.
