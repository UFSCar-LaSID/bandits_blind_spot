
LIN = {

}

LIN_UCB_HYPERPARAMETERS = {
    'alpha': [0.1, 0.5, 1, 1.5, 2]
}

LIN_GREEDY_HYPERPARAMETERS = {
    'epsilon': [0.01, 0.05, 0.1, 0.25, 0.5]
}

LIN_TS_HYPERPARAMETERS = {
    'alpha': [0.1, 0.5, 1, 1.5, 2]
}

TIME_AWARE_LIN_BOLTZMANN_HYPERPARAMETERS = {
    'rho': [1], # Increase or decrease time importance
    'T_fixed': [0.1], # Temperature fixed - If you don't want to use adpatative temperature (time-aware)
    'temperature_method': ["traditional"],
    'alpha': [0.1],
    'H_min': [0.5],
    'H_max': [1.5],
    'use_variance_entropy': [True]
}


CLUSTERS_LIN_HYPERPARAMETERS = {
    'neighborhood_policy' : [
        {}
    ]
}

CLUSTERS_LIN_UCB_HYPERPARAMETERS = {
    'neighborhood_policy' : [
        {}
    ],
    'learning_policy' : [
        {
            'alpha': 0.1
        },
        {
            'alpha': 0.5
        },
        {
            'alpha': 1
        },
        {
            'alpha': 1.5
        },
        {
            'alpha': 2
        }
    ]
}

CLUSTERS_LIN_GREEDY_HYPERPARAMETERS = {
    'neighborhood_policy' : [
        {}
    ],
    'learning_policy' : [
        {
            'epsilon': 0.01
        },
        {
            'epsilon': 0.05
        },
        {
            'epsilon': 0.1
        },
        {
            'epsilon': 0.25
        },
        {
            'epsilon': 0.5
        }
    ]
}

CLUSTERS_LIN_TS_HYPERPARAMETERS = {
    'neighborhood_policy' : [
        {}
    ],
    'learning_policy' : [
        {
            'alpha': 0.1
        },
        {
            'alpha': 0.5
        },
        {
            'alpha': 1
        },
        {
            'alpha': 1.5
        },
        {
            'alpha': 2
        }
    ]
}