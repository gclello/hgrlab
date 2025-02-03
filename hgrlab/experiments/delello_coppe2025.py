import os
import numpy as np

from ..utils import AssetManager
from ..models.hgrdtw import k_fold_cost, FeatureSet
from ..models.hgrdtw import build_classifier, fit, predict

from ..experiments.data import emgepn10
from ..experiments import run_experiments, download_assets
from .functions import eval_hgr_systems, tune_seg_thresholds

def tune_segmentation_threshold(config):
    threshold_min = config['threshold_min']
    threshold_max = config['threshold_max']
    threshold_direction = config['threshold_direction']

    if 'cv_options' in config.keys():
        cv_options = config['cv_options']
    else:
        cv_options = None

    if 'classifier_options' in config.keys():
        classifier_options = config['classifier_options']
    else:
        classifier_options = None

    if threshold_direction == 'desc':
        thresholds = np.flip(np.arange(threshold_min, threshold_max+1))
    else:
        thresholds = np.arange(threshold_min, threshold_max+1)

    HUGE_ERROR = 1000000
    thresholds_errors = np.full(np.size(thresholds), HUGE_ERROR, dtype=np.uint32)
    thresholds_predictions = np.zeros((np.size(thresholds)),dtype=np.uint32)

    feature_set_config = config['feature_set_config']

    for threshold_id, threshold in enumerate(thresholds):
        feature_set_config['activity_threshold'] = threshold

        errors, predictions = k_fold_cost(
            feature_set_config=config['feature_set_config'],
            folds=config['cv_folds'],
            classifier_name=config['classifier_name'],
            cv_options=cv_options,
            classifier_options=classifier_options,
        )

        thresholds_errors[threshold_id] = errors
        thresholds_predictions[threshold_id] = predictions

        if errors == 0:
            break

    optimum_index = np.argmin(thresholds_errors)

    return {
        'threshold': thresholds[optimum_index],
        'errors': thresholds_errors[optimum_index],
        'predictions': thresholds_predictions[optimum_index],
    }

def majority_voting(
    predictions,
    window_start_activity_indexes,
    window_end_activity_indexes,
    window_length,
):
    has_full_muscle_contraction = np.logical_and(
        window_start_activity_indexes != 0,
        window_end_activity_indexes != window_length,
    )

    predictions[has_full_muscle_contraction == False] = 'relax'
    
    unique, counts = np.unique(predictions, return_counts=True)
    non_relax_unique_predictions = unique[unique != 'relax']
    non_relax_predictions_count = counts[unique != 'relax']

    if len(non_relax_unique_predictions) > 0:
        reduced_prediction = non_relax_unique_predictions[non_relax_predictions_count.argmax()]
    else:
        reduced_prediction = 'relax'
    
    return reduced_prediction

def eliminate_consecutive_gestures(
    predictions,
    window_start_activity_indexes,
    window_end_activity_indexes,
    window_length,
):
    reduced_prediction = 'relax'

    has_full_muscle_contraction = np.logical_and(
        window_start_activity_indexes != 0,
        window_end_activity_indexes != window_length,
    )

    predictions[has_full_muscle_contraction == False] = 'relax'
    predictions[0] = 'relax'
    predictions[predictions == np.roll(predictions, 1)] = 'relax'

    unique_labels = np.unique(predictions[predictions != 'relax'])

    label_names = [
        'relax',
        'fist',
        'wave_in',
        'wave_out',
        'fingers_spread',
        'double_tap',
    ]

    if np.size(unique_labels) > 0:
        label_ids = np.vectorize(lambda label:  label_names.index(label) + 1)(unique_labels)
        order = np.argsort(label_ids)
        ordered_unique_labels = unique_labels[order]
        reduced_prediction = ordered_unique_labels[0]

    return reduced_prediction

def eval_hgr_system(config):
    experiments = config['experiments']
    classifier_name = config['classifier_name']

    if 'classifier_options' in config.keys():
        classifier_options = config['classifier_options']
    else:
        classifier_options = None

    if 'prediction_reduction_method' in config.keys():
        reduction_method = config['prediction_reduction_method']
    else:
        reduction_method = None

    reduce_predictions = majority_voting if reduction_method == 'majority_voting' else eliminate_consecutive_gestures

    config['feature_set_config']['ds_type'] = 'training'
    fs_training = FeatureSet.build_and_extract(config['feature_set_config'])
    X_train = fs_training.get_data('dtw')
    y_train = fs_training.get_data('labels')

    config['feature_set_config']['ds_type'] = 'test'
    fs_test = FeatureSet.build_and_extract(config['feature_set_config'])
    X_test = fs_test.get_data('dtw')
    y_test = fs_test.get_data('labels')
    test_activity_indices = fs_test.get_data('predicted_indices')

    test_trials = X_test.shape[0]
    test_window_length = X_test.shape[2]

    errors = np.zeros((experiments),dtype=np.uint32)
    trials = np.zeros((experiments),dtype=np.uint32)

    for experiment in np.arange(0, experiments):
        model = build_classifier(classifier_name, classifier_options)
        fit(model, X_train, y_train)

        prediction = np.full((test_trials,), 'relax', dtype='U14')
        
        for trial_id, X_test_windows in enumerate(X_test):
            test_window_predictions = predict(model, X_test_windows)

            prediction[trial_id] = reduce_predictions(
                test_window_predictions,
                test_activity_indices[trial_id,:,0],
                test_activity_indices[trial_id,:,1],
                test_window_length,
            )

        errors[experiment] = np.size(y_test[y_test != prediction])
        trials[experiment] = np.size(y_test)

    return {
        'errors': errors,
        'predictions': trials,
    }

def tune_and_eval_hgr_systems_by_classifier_and_user(
    dataset_name,
    ds_dir,
    fs_dir,
    user_ids,
    options,
):
    tuning_result = tune_seg_thresholds.run(
        dataset_name,
        ds_dir,
        fs_dir,
        user_ids,
        options,
        threshold_direction='desc',
        threshold_min=10,
        threshold_max=20,
        tune_segmentation_threshold=tune_segmentation_threshold,
    )

    classifiers_thresholds = {}

    for i, classifier in enumerate(options['classifier_names']):
        classifiers_thresholds[classifier] = tuning_result['data'][i]

    options['thresholds'] = classifiers_thresholds

    eval_result = eval_hgr_systems.run(
        dataset_name,
        ds_dir,
        fs_dir,
        user_ids,
        options,
        experiment_runs=100,
        eval_hgr_system=eval_hgr_system,
    )

    eval_result['message'] = '{TUNING}\n\n{EVAL}'.format(
        TUNING=tuning_result['message'],
        EVAL=eval_result['message'],
    )

    return eval_result

def main():
    publication = "A Comparative Study of Classifiers for sEMG-Based Hand Gesture Recognition Systems"
    author = 'Guilherme C. De Lello'

    title = '{PUBLICATION}\nExperiments conducted by {AUTHOR} on February 2025'.format(
        PUBLICATION=publication,
        AUTHOR=author,
    )

    experiment = 'coppe2025'
    dataset_name='emgepn10'
    dtw_impl = 'dtaidistance'
    cv_strategy = '5-fold-stratified-cv'
    prediction_reduction_method = 'majority_voting'

    if cv_strategy == '5-fold-stratified-cv':
        cv_folds = 5
        cv_options = None
    else:
        cv_folds = 4
        cv_options = {'val_size_per_class': 2}  

    base_dir = os.path.join(
        AssetManager.get_base_dir(),
        '%s_%s_%s' % (experiment, dataset_name, dtw_impl)
    )

    ds_dir = '%s_%s' % (base_dir, "ds")
    fs_dir = '%s_%s' % (base_dir, "fs")

    assets = {
        **emgepn10.get_dataset_assets('training'),
        **emgepn10.get_dataset_assets('test'),
    }

    def setup():
        download_assets(AssetManager(), assets, ds_dir)

    run_experiments(
        title=title,
        dataset_name=dataset_name,
        ds_dir=ds_dir,
        fs_dir=fs_dir,
        user_ids=np.arange(1, 11),
        setup=setup,
        experiments=[
            tune_and_eval_hgr_systems_by_classifier_and_user,
        ],
        options={
            'cv_folds': cv_folds,
            'cv_options': cv_options,
            'classifier_names': [
                'svm',
                'lr',
                'lda',
                'knn',
                'dt',
            ],
            'dtw_impl': dtw_impl,
            'feature_window_length': 500,
            'feature_overlap_length': 490,
            'prediction_reduction_method': prediction_reduction_method,
        }
    )

if __name__ == '__main__':
    main()
