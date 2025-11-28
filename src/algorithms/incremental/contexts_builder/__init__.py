
import pandas as pd
import src

from src.algorithms.incremental.contexts_builder.UserContextsBuilder import UserContextsBuilder
from src.algorithms.incremental.contexts_builder.ItemsMeanContextsBuilder import ItemsMeanContextsBuilder
from src.algorithms.incremental.contexts_builder.ItemsConcatContextsBuilder import ItemsConcatContextsBuilder
from src.algorithms.incremental.contexts_builder.CombinationContextBuilder import ItemsConcatUserContextsBuilder, ItemsConcatItemsMeanUserContextsBuilder, ItemsConcatItemsMeanContextsBuilder, ItemsMeanUserContextsBuilder
from src.algorithms.incremental.contexts_builder.hyperparameters import ITEMS_CONCAT_HYPERPARAMETERS, ITEMS_MEAN_HYPERPARAMETERS, USER_HYPERPARAMETERS, ITEMS_CONCAT_USER_HYPERPARAMETERS, ITEMS_MEAN_USER_HYPERPARAMETERS, ITEMS_CONCAT_ITEMS_MEAN_HYPERPARAMETERS, ITEMS_CONCAT_ITEMS_MEAN_USER_HYPERPARAMETERS


CONTEXTS_BUILDER_TABLE = pd.DataFrame(
    [[1,  'user',                        UserContextsBuilder,                        USER_HYPERPARAMETERS],
     [2,  'item_concat',                 ItemsConcatContextsBuilder,                 ITEMS_CONCAT_HYPERPARAMETERS],
     [3,  'item_mean',                   ItemsMeanContextsBuilder,                   ITEMS_MEAN_HYPERPARAMETERS],
     [4,  'item_concat+user',            ItemsConcatUserContextsBuilder,             ITEMS_CONCAT_USER_HYPERPARAMETERS],
     [5,  'item_mean+user',              ItemsMeanUserContextsBuilder,               ITEMS_MEAN_USER_HYPERPARAMETERS],
     [6,  'item_concat+item_mean',       ItemsConcatItemsMeanContextsBuilder,        ITEMS_CONCAT_ITEMS_MEAN_HYPERPARAMETERS],
     [7,  'item_concat+item_mean+user',  ItemsConcatItemsMeanUserContextsBuilder,    ITEMS_CONCAT_ITEMS_MEAN_USER_HYPERPARAMETERS]],
    columns=[src.CONTEXTS_BUILDER_ID, src.CONTEXTS_BUILDER_NAME, src.CONTEXTS_BUILDER_CLASS, src.CONTEXTS_BUILDER_HYPERPARAMETERS]
).set_index(src.CONTEXTS_BUILDER_ID)
