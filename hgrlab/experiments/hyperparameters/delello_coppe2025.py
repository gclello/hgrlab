import numpy as np

def generate_svm_options(log_samples=10):
    kernels = ['linear', 'rbf', 'poly', 'sigmoid']
    
    regularizations = np.logspace(-4, 4, log_samples)
    gammas = np.logspace(-4, 4, log_samples)
    default_tolerance = 0.001
    default_cache_size = 8192

    options = []

    for kernel in kernels:
        for regularization in regularizations:
            if kernel == 'linear':
                options.append({
                    'kernel': kernel,
                    'C': regularization,
                    'tol': default_tolerance,
                    'cache_size': default_cache_size,
                })
            
            else:    
                for gamma in gammas:
                    options.append({
                        'kernel': kernel,
                        'C': regularization,
                        'gamma': gamma,
                        'tol': default_tolerance,
                        'cache_size': default_cache_size,
                    })

    return options

def generate_lr_options(log_samples=10):
    solvers = [
        'liblinear',
        'lbfgs',
        'newton-cg',
        'newton-cholesky',
        'sag',
        'saga'
    ]

    penalties = {
        'lbfgs': ['l2'],
        'liblinear': ['l2', 'l1'],
        'newton-cg': ['l2'],
        'newton-cholesky': ['l2'],
        'sag': ['l2'],
        'saga': ['elasticnet', 'l2', 'l1'],
    }

    regularizations = np.logspace(-4, 4, log_samples)
    default_tolerance = 0.0001
    default_max_iterations = 200

    options = []

    for solver in solvers:
        for penalty in penalties[solver]:
            for regularization in regularizations:
                option = {
                    'solver': solver,
                    'penalty': penalty,
                    'C': regularization,
                    'tol': default_tolerance,
                    'max_iter': default_max_iterations,
                }
                
                if penalty == 'elasticnet':
                    option['l1_ratio'] = 0.5
                
                options.append(option)

    return options

def generate_lda_options():
    solvers = ['svd', 'lsqr']
    default_tolerance = 0.0001
    
    options = []
    
    for solver in solvers:
        options.append({
            'solver': solver,
            'tol': default_tolerance,
        })
    
    return options

def generate_knn_options():
    neighbors = np.arange(1, 11)
    weights = ['uniform', 'distance']
    algorithms = ['ball_tree', 'kd_tree', 'brute']
    metrics = ['euclidean']
    
    options = []
    
    for neighbor in neighbors:
        for weight in weights:
            for algorithm in algorithms:
                for metric in metrics:                    
                    options.append({
                        'n_neighbors': neighbor,
                        'weights': weight,
                        'algorithm': algorithm,
                        'metric': metric,
                    })
    
    return options

def generate_dt_options():
    criteria = ['gini', 'entropy', 'log_loss']
    options = []
    
    for criterion in criteria:
        options.append({
            'criterion': criterion,
        })
    
    return options

def generate_twsd_options():
    address_sizes = np.arange(4, 49, 4)
    thermometer_size_powers = np.arange(3, 15)
    
    options = []
    
    for address_size in address_sizes:
        for thermometer_size_power in thermometer_size_powers:
            options.append({
                'address_size': address_size,
                'thermometer_size': 2**thermometer_size_power,
            })
    
    return options
