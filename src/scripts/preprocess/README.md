<div align="center">
    
Language

English | [Portuguese (BR)](./README_PT-BR.md)
    
</div>

# Datasets Preprocessing

You need to download and preprocess the datasets to run subsequent experiments.

The preprocessing code removes all **repeated interactions** from the datasets (those with identical item, user, and interaction timestamp). Only the first interaction of a set of repeated interactions is kept, leaving no duplicates in the datasets. Additionally, all interactions with **missing data** are deleted. The column names of the interaction tables are standardized so that all databases can be loaded in the same way later (with the same code). Finally, the **timestamp field** is generated based on the datetime column if this information does not exist, and vice versa.

## Databases

Install the datasets in the indicated locations to preprocess the datasets. Below is a list of databases that can be downloaded and preprocessed:

- [AmazonBeauty](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/All_Beauty.jsonl.gz): place the `All_Beauty.jsonl` file in `raw/amazon-beauty`
- [AmazonBooks](https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews): download and extract into `raw/amazon-books`
- [BestBuy](https://www.kaggle.com/c/acm-sf-chapter-hackathon-big/data?select=train.csv): place the `train.csv` file in `raw/BestBuy`
- [Delicious2K](https://grouplens.org/datasets/hetrec-2011): download `hetrec2011-delicious-2k.zip` from the `Delicious Bookmarks` section and extract it into `raw/delicious2k`
- [MovieLens-100K](https://grouplens.org/datasets/movielens/): download `ml-100k.zip` from the `MovieLens 100K Dataset` section and extract it into `raw/ml-100k`
- [MovieLens-25M](https://grouplens.org/datasets/movielens/): download `ml-25m.zip` from the `MovieLens 25M Dataset` section and extract it into `raw/ml-25m`

## Running the Preprocessing Code

Once the datasets are installed, you need to run the preprocessing code to use them for training and recommendation generation.
To do this, execute the following command:

```
python src/scripts/preprocess/main.py
```

Running it this way will prompt you to enter the datasets to be preprocessed. Type the indices of the databases you want to preprocess, separated by spaces.

Another way to choose which databases to process is by providing the `--datasets` argument when executing the algorithm.

The command to use is:

```
python src/scripts/preprocess.py --datasets <datasets>
```

Replace `<datasets>` with the names (or indices) separated by spaces. The available databases for preprocessing are:

- \[1\]: amazon-beauty
- \[2\]: amazon-books
- \[3\]: amazon-games
- \[4\]: bestbuy
- \[5\]: delicious2k
- \[6\]: delicious2k-urlPrincipal
- \[7\]: ml-100k
- \[8\]: ml-25m
- \[9\]: retailrocket
- all (will use all datasets)
