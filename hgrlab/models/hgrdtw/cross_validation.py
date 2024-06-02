import math
import numpy as np

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

def balanced_train_val_split(X, y, total_folds, fold=0):
    validation_indices = get_balanced_validation_indices(
        y,
        total_folds=total_folds,
        fold=fold,
    )

    X_train = X[validation_indices == False]
    y_train = y[validation_indices == False]

    X_val = X[validation_indices == True]
    y_val = y[validation_indices == True]

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

def get_balanced_validation_indices(labels, total_folds=5, fold=0):
    unique_labels = np.unique(labels)
    validation_indices = []
    
    for label in unique_labels:
        label_indices = np.argwhere(labels == label).flatten()
        samples_in_class = len(label_indices)
        validation_samples = math.ceil(samples_in_class/total_folds)
        offset = validation_samples * fold
        label_validation_indices = np.roll(label_indices, -offset)[0:validation_samples]
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
                total_folds=folds,
                fold=fold,
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
