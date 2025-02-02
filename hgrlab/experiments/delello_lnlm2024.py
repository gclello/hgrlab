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

    HUGE_ERROR = 1000
    thresholds_errors = np.full(np.size(thresholds), HUGE_ERROR)

    for threshold_id, threshold in enumerate(thresholds):
        config['feature_set_config']['activity_threshold'] = threshold

        errors, _ = k_fold_cost(
            feature_set_config=config['feature_set_config'],
            folds=config['cv_folds'],
            classifier_name=config['classifier_name'],
            cv_options=cv_options,
            classifier_options=classifier_options,
        )

        thresholds_errors[threshold_id] = errors

        if errors == 0:
            break

    return thresholds[np.argmax(thresholds_errors)]

def reduce_window_predictions(
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
        model = build_classifier(classifier_name)
        fit(model, X_train, y_train)

        prediction = np.full((test_trials,), 'relax', dtype='U14')
        
        for trial_id, X_test_windows in enumerate(X_test):
            test_window_predictions = predict(model, X_test_windows)

            prediction[trial_id] = reduce_window_predictions(
                test_window_predictions,
                test_activity_indices[trial_id,:,0],
                test_activity_indices[trial_id,:,1],
                test_window_length,
            )

        errors[experiment] = np.size(y_test[y_test != prediction])
        trials[experiment] = np.size(y_test)

    return errors, trials

def tune_segmentation_thresholds_by_classifier_and_user(
    experiment_id,
    total_experiments,
    dataset_name,
    assets_dir,
    user_ids,
    options,
):
    return tune_seg_thresholds.run(
        experiment_id,
        total_experiments,
        dataset_name,
        assets_dir,
        user_ids,
        options,
        threshold_min=10,
        threshold_max=20,
        tune_segmentation_threshold=tune_segmentation_threshold,
    )

def eval_hgr_systems_by_classifier_and_user(
    experiment_id,
    total_experiments,
    dataset_name,
    assets_dir,
    user_ids,
    options,
):
    return eval_hgr_systems.run(
        experiment_id,
        total_experiments,
        dataset_name,
        assets_dir,
        user_ids,
        options,
        experiment_runs=100,
        eval_hgr_system=eval_hgr_system,
    )

def main():
    authors = 'Guilherme C. De Lello, Gabriel S. Chaves, Juliano F. Caldeira, and Markus V.S. Lima'
    title = 'HGR experiments conducted by %s on March 2024' % authors

    def setup():
        assets = {
            **emgepn10.get_dataset_assets('training'),
            **emgepn10.get_feature_assets('test'),
        }

        download_assets(AssetManager(), assets)

    run_experiments(
        title,
        dataset_name='emgepn30',
        assets_dir=AssetManager.get_base_dir(),
        user_ids=np.arange(1, 11),
        setup=setup,
        experiments=[
            tune_segmentation_thresholds_by_classifier_and_user,
            eval_hgr_systems_by_classifier_and_user,
        ],
        options={
            'cv_folds': 4,
            'cv_options': {'val_size_per_class': 2},
            'classifier_names': [
                'svm',
                'lr',
                'lda',
                'knn',
                'dt',
            ],
            'thresholds': {
                'svm': [19, 20, 12, 19, 19, 19, 19, 19, 19, 19],
                'lr': [19, 20, 19, 19, 19, 19, 19, 19, 19, 19],
                'lda': [19, 20, 15, 19, 17, 16, 19, 19, 19, 19],
                'knn': [19, 20, 19, 19, 14, 16, 19, 19, 19, 19],
                'dt': [18, 18, 20, 16, 11, 18, 14, 19, 19, 20],
            },
            'feature_window_length': 500,
            'feature_overlap_length': 490,
            'dtw_impl': 'fastdtw',
        }
    )

if __name__ == '__main__':
    main()
