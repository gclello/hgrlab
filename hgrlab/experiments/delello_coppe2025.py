import os
import numpy as np

from ..utils import AssetManager, plot_radar
from ..models.hgrdtw import k_fold_cost, FeatureSet
from ..models.hgrdtw import build_classifier, fit, predict

from .data import emgepn10
from . import run_experiments, download_assets
from .functions import eval_hgr_systems, tune_seg_thresholds, tune_hyperparams
from .hyperparameters import delello_coppe2025 as hypeparams

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

    optimal_index = np.argmin(thresholds_errors)

    return {
        'threshold': thresholds[optimal_index],
        'errors': thresholds_errors[optimal_index],
        'predictions': thresholds_predictions[optimal_index],
    }

def get_k_fold_cost(config):
    errors, predictions = k_fold_cost(
        feature_set_config=config['feature_set_config'],
        folds=config['cv_folds'],
        classifier_name=config['classifier_name'],
        cv_options=config['cv_options'],
        classifier_options=config['classifier_options'],
    )
    
    return {
        'errors': errors,
        'predictions': predictions,
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
    out_dir,
    user_ids,
    options,
    default_experiment_runs=1,
    default_min_theshold=10,
    default_max_theshold=20,
    default_tune_hyperparams_skip=False,
    default_tune_hyperparams_random=False,
    default_eval_experiment_runs=100,
):
    classifier_names = options['classifier_names']
    number_of_classifiers = np.size(classifier_names)
    number_of_users = np.size(user_ids)

    if 'pipeline' in options.keys():
        pipeline_options = options['pipeline']
        experiment_runs = pipeline_options['experiment_runs']
        threshold_min = pipeline_options['tune_seg_threshold']['threshold_min']
        threshold_max = pipeline_options['tune_seg_threshold']['threshold_max']
        skip_hyperparams_tuning = pipeline_options['tune_hyperparams']['skip']
        randomize_hyperparams_tuning = pipeline_options['tune_hyperparams']['random']
        eval_experiment_runs = pipeline_options['eval']['experiment_runs']
    else:
        experiment_runs = default_experiment_runs
        threshold_min = default_min_theshold
        threshold_max = default_max_theshold
        skip_hyperparams_tuning = default_tune_hyperparams_skip
        randomize_hyperparams_tuning = default_tune_hyperparams_random
        eval_experiment_runs = default_eval_experiment_runs

    accuracy = np.zeros((
        experiment_runs,
        number_of_classifiers,
        number_of_users,
        eval_experiment_runs,
    ))

    output_list = []

    for experiment_id in np.arange(0, experiment_runs):
        classifier_options = {
            'svm': hypeparams.generate_svm_options(),
            'lr': hypeparams.generate_lr_options(),
            'lda': hypeparams.generate_lda_options(),
            'knn': hypeparams.generate_knn_options(),
            'dt': hypeparams.generate_dt_options(),
            'twsd': hypeparams.generate_twsd_options(),
        }

        best_hyperparameters = {}
        best_hyperparameters_messages = []
        best_seg_thresholds_messages = []

        if not skip_hyperparams_tuning:
            if randomize_hyperparams_tuning:
                rng = np.random.default_rng()
                for i, classifier_name in enumerate(classifier_names):
                    if classifier_name not in classifier_options.keys():
                        continue

                    random_indices = rng.integers(
                        low=0,
                        high=np.size(classifier_options[classifier_name]),
                        size=number_of_users,
                    )

                    best_hyperparameters[classifier_name] = {}
                    for i, user_id in enumerate(user_ids):
                        best_hyperparameters[classifier_name][user_id] = classifier_options[classifier_name][random_indices[i]]
            else:
                seg_tuning_result1 = tune_seg_thresholds.run(
                    dataset_name,
                    ds_dir,
                    fs_dir,
                    user_ids,
                    options,
                    threshold_direction='desc',
                    threshold_min=threshold_min,
                    threshold_max=threshold_max,
                    tune_segmentation_threshold=tune_segmentation_threshold,
                )

                best_seg_thresholds_messages.append(seg_tuning_result1['message'])
                classifier_thresholds = {}

                for i, classifier in enumerate(classifier_names):
                    classifier_thresholds[classifier] = seg_tuning_result1['data'][i]

                options['thresholds'] = classifier_thresholds

                for i, classifier_name in enumerate(classifier_names):
                    if classifier_name not in classifier_options.keys():
                        continue
                    elif len(classifier_options[classifier_name]) == 0:
                        continue
                    elif len(classifier_options[classifier_name]) == 1:
                        single_option = classifier_options[classifier_name][0]
                        best_hyperparameters[classifier_name] = {}
                        for i, user_id in enumerate(user_ids):
                            best_hyperparameters[classifier_name][user_id] = single_option
                        continue

                    options['classifier_name'] = classifier_name
                    options['classifier_options_list'] = classifier_options[classifier_name]

                    hyperparams_tuning_result = tune_hyperparams.run(
                        dataset_name,
                        ds_dir,
                        fs_dir,
                        user_ids,
                        options,
                        cost_function=get_k_fold_cost,
                    )

                    best_options = hyperparams_tuning_result['data']['best_options']
                    best_hyperparameters[classifier_name] = best_options
                    best_hyperparameters_messages.append(
                        hyperparams_tuning_result['message']
                    )

            options['classifier_options'] = best_hyperparameters

        seg_tuning_result2 = tune_seg_thresholds.run(
            dataset_name,
            ds_dir,
            fs_dir,
            user_ids,
            options,
            threshold_direction='desc',
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            tune_segmentation_threshold=tune_segmentation_threshold,
        )

        best_seg_thresholds_messages.append(seg_tuning_result2['message'])
        classifier_thresholds = {}

        for i, classifier in enumerate(classifier_names):
            classifier_thresholds[classifier] = seg_tuning_result2['data'][i]

        options['thresholds'] = classifier_thresholds
        
        eval_result = eval_hgr_systems.run(
            dataset_name,
            ds_dir,
            fs_dir,
            user_ids,
            options,
            experiment_runs=eval_experiment_runs,
            eval_hgr_system=eval_hgr_system,
        )

        eval_result['message'] = '{HYPER_TUNING}\n\n{SEG_TUNING}\n\n{EVAL}'.format(
            HYPER_TUNING='\n\n'.join(best_hyperparameters_messages),
            SEG_TUNING='\n\n'.join(best_seg_thresholds_messages),
            EVAL=eval_result['message'],
        )

        output_list.append(eval_result['message'])

        accuracy[experiment_id,:,:,:] = eval_result['data'][:,:,:]

    output_list.append('## Mean accuracy')
    output_list.append(repr(accuracy.mean(axis=(0,2,3))))
    output_list.append('## Std')
    output_list.append(repr(accuracy.std(axis=(0,2,3), ddof=1)))

    accuracy_per_classifier = accuracy.mean(axis=(0,2,3))
    accuracy_per_experiment = accuracy.mean(axis=2)
    std_per_experiment = accuracy_per_experiment.std(ddof=1, axis=(0,2))

    save_radar_plot(
        dataset_name,
        classifier_names,
        accuracy_per_classifier,
        std_per_experiment,
        out_dir,
    )

    return {'message': '\n\n'.join(output_list)}

def save_radar_plot(
    dataset_name,
    classifier_names,
    values,
    std,
    out_dir,
):
    if out_dir is None:
        return
    
    config = {
        'emgepn10': {
            'ticks': [86, 88, 90, 92, 94, 96],
            'range_min': 85,
            'range_max': 97,
        },
        'emgepn120': {
            'ticks': [96, 96.5, 97, 97.5, 98],
            'range_min': 86,
            'range_max': 98.2,
        },
    }

    titles = {
        'svm': 'Support Vector Machine',
        'lr': 'Logistic Regression',
        'lda': 'Linear Discriminant Analysis',
        'knn': '<i>K</i>-Nearest Neighbors',
        'dt': 'Decision Tree',
        'twsd': 'WiSARD',
    }

    effective_titles = []
    for classifier in classifier_names:
        if classifier in titles.keys():
            effective_titles.append(titles[classifier])

    plot_radar(
        effective_titles,
        values*100,
        std*100,
        ticks=config[dataset_name]['ticks'],
        range_min=config[dataset_name]['range_min'],
        range_max=config[dataset_name]['range_max'],
        output_path=os.path.join(out_dir, '%s_radar.png' % dataset_name),
    )

def main():
    publication = "A Comparative Study of Classifiers for sEMG-Based Hand Gesture Recognition Systems"
    author = 'Guilherme C. De Lello'

    title = '{PUBLICATION}\nExperiments conducted by {AUTHOR} on February 2025'.format(
        PUBLICATION=publication,
        AUTHOR=author,
    )

    profiles = {
        'coppe2025': {
            'dtw_impl': 'dtaidistance',
            'cv_folds': 5,
            'cv_options': None,
            'prediction_reduction_method': 'majority_voting',
            'feature_window_length': 500,
            'feature_overlap_length': 490,
            'classifier_names': [
                'svm',
                'lr',
                'lda',
                'knn',
                'dt',
                'twsd',
            ],
            'pipeline': {
                'experiment_runs': 1,
                'tune_seg_threshold': {
                    'threshold_min': 10,
                    'threshold_max': 20,
                },
                'tune_hyperparams': {
                    'skip': False,
                    'random': False,
                },
                'eval': {
                    'experiment_runs': 100,
                },
            },
        },
        'lnlm2024': {
            'dtw_impl': 'fastdtw',
            'cv_folds': 4,
            'cv_options': {'val_size_per_class': 2},
            'prediction_reduction_method': 'baseline',
            'feature_window_length': 500,
            'feature_overlap_length': 490,
            'classifier_names': [
                'svm',
                'lr',
                'lda',
                'knn',
            ],
        }
    }

    datasets = {
        'emgepn10': {
            'user_ids': np.arange(1, 11),
            'assets': {
                'semg': {
                    **emgepn10.get_dataset_assets('training'),
                    **emgepn10.get_dataset_assets('test'),
                },
                'feature': {
                    **emgepn10.get_feature_assets('test')
                }
            }
        },
        'emgepn120': {
            'user_ids': np.arange(1, 61),
            'assets': {
                'semg': {},
                'feature': {},
            },
        }
    }

    try:
        from dotenv import dotenv_values
        config = dotenv_values(".env")
    except(Exception):
        config = {}

    profile = config['PROFILE'] if 'PROFILE' in config.keys() else 'coppe2025'
    ds_name = config['DS_NAME'] if 'DS_NAME' in config.keys() else 'emgepn10'
    out_dir = config['OUT_DIR'] if 'OUT_DIR' in config.keys() else None

    base_dir = os.path.join(
        AssetManager.get_base_dir(),
        '%s_%s_%s' % (profile, ds_name, profiles[profile]['dtw_impl'])
    )

    ds_dir = config['DS_DIR'] if 'DS_DIR' in config.keys() else None
    fs_dir = config['FS_DIR'] if 'FS_DIR' in config.keys() else '%s_%s' % (base_dir, 'fs')

    should_download_semg_assets = False

    if ds_dir is None:
        should_download_semg_assets = True
        ds_dir = '%s_%s' % (base_dir, 'ds')

    def setup():
        semg_assets = datasets[ds_name]['assets']['semg']
        feature_assets = datasets[ds_name]['assets']['feature']
        
        if should_download_semg_assets and semg_assets is not None:
            download_assets(AssetManager(), semg_assets, ds_dir)

        if feature_assets is not None:
            download_assets(AssetManager(), feature_assets, fs_dir)

    print('Dataset name = %s ("%s")' % (ds_name, ds_dir))
    print('Temp path = "%s"' % fs_dir)
    print('Options = %s' % profiles[profile])

    run_experiments(
        title=title,
        dataset_name=ds_name,
        ds_dir=ds_dir,
        fs_dir=fs_dir,
        user_ids=datasets[ds_name]['user_ids'],
        setup=setup,
        experiments=[
            tune_and_eval_hgr_systems_by_classifier_and_user,
        ],
        options=profiles[profile],
        out_dir=out_dir,
    )

if __name__ == '__main__':
    main()
