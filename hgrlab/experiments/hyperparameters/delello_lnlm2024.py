import numpy as np

def generate_svm_options():
    return [{
        'kernel': 'rbf',
        'C': 1.0,
        'gamma': 'auto',
        'tol': 0.001,
        'cache_size': 8192,
    }]

def generate_lr_options():
    return [{
        'solver': 'liblinear',
        'penalty': 'l2',
        'C': 1.0,
        'tol': 0.0001,
        'max_iter': 200,
    }]

def generate_lda_options():
    return [{
        'solver': 'svd',
        'tol': 0.0001,
    }]

def generate_knn_options():
    return [{
        'n_neighbors': 5,
        'weights': 'uniform',
        'metric': 'euclidean',
    }]

def generate_dt_options():
    return [{
        'criterion': 'gini',
    }]

def generate_twsd_options():
    return [{
        'address_size': 4,
        'thermometer_size': 2**3,
    }]

