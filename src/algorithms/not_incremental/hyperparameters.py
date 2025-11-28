ALS_HYPERPARAMETERS = {
    'negative_strategy': ['confidence', 'binary', 'discard' , None],
    'factors': [4, 8, 16, 32],
    'regularization': [0.001, 0.01, 0.1],
    'iterations': [5, 15, 30],
}


BPR_HYPERPARAMETERS = {
    'negative_strategy': ['discard' , None],
    'factors': [4, 8, 16, 32],
    'learning_rate': [0.001, 0.01, 0.1],
    'regularization': [0.001, 0.01, 0.1],
    'iterations': [50, 100, 150],
}