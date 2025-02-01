import math
import numpy as np

from sklearn.model_selection import StratifiedKFold

from .feature_set import FeatureSet
from .classification import build_classifier, fit, predict

def fixed_train_val_split(X, y, val_size_per_class, offset=0):
    validation_indices = get_fixed_validation_indices(
        y,
        limit_per_class=val_size_per_class,
        offset=offset,
    )

    X_train = X[validation_indices == False]
    y_train = y[validation_indices == False]

    X_val = X[validation_indices == True]
    y_val = y[validation_indices == True]

    return X_train, y_train, X_val, y_val

def balanced_train_val_split(X, y, folds, fold=0, shuffle=None, random_state=None):
    skf = StratifiedKFold(n_splits=folds, shuffle=shuffle, random_state=random_state)

    train_indices, val_indices = [(train, test) for (train, test) in skf.split(X, y)][fold]

    X_train = X[train_indices]
    y_train = y[train_indices]

    X_val = X[val_indices]
    y_val = y[val_indices]

    return X_train, y_train, X_val, y_val

def get_fixed_validation_indices(labels, limit_per_class=2, offset=0):
    unique_labels = np.unique(labels)
    validation_indices = []
    
    for label in unique_labels:
        label_indices = np.argwhere(labels == label).flatten()
        label_validation_indices = np.roll(label_indices, -offset)[0:limit_per_class]
        validation_indices = validation_indices + label_validation_indices.tolist()
    
    is_validation = np.full(len(labels), False)
    is_validation[validation_indices] = True
        
    return is_validation

def k_fold_cost(
    feature_set_config,
    folds,
    classifier_name,
    cv_options=None,
    classifier_options=None,
):
    if cv_options is not None and 'val_size_per_class' in cv_options:
        val_size_per_class = cv_options['val_size_per_class']
    else:
        val_size_per_class = None

    if cv_options is not None and 'shuffle' in cv_options:
        shuffle = cv_options['shuffle']
    else:
        shuffle = False

    if cv_options is not None and 'random_state' in cv_options:
        random_state = cv_options['random_state']
    else:
        random_state = None
    
    fs = FeatureSet.build_and_extract(feature_set_config)
    features = fs.get_data('dtw')
    labels = fs.get_data('labels')
    
    total_errors = 0
    total_predictions = 0
    
    for fold in np.arange(0, folds):
        if val_size_per_class is not None:
            X_train, y_train, X_val, y_val = fixed_train_val_split(
                X=features,
                y=labels,
                val_size_per_class=val_size_per_class,
                offset=fold,
            )
        else:
            X_train, y_train, X_val, y_val = balanced_train_val_split(
                X=features,
                y=labels,
                folds=folds,
                fold=fold,
                shuffle=shuffle,
                random_state=random_state,
            )
    
        model = build_classifier(classifier_name, classifier_options)
        fit(model, X_train, y_train)
        predictions = predict(model, X_val)
    
        number_of_predictions = np.size(y_val)
        correct_predictions = np.count_nonzero(predictions == y_val)
        errors = number_of_predictions - correct_predictions

        total_predictions = total_predictions + number_of_predictions
        total_errors = total_errors + errors
    
    return total_errors, total_predictions
