import numpy as np
import multiprocessing as mp
import concurrent.futures
import datetime

from ..models.hgrdtw import k_fold_cost, FeatureSet
from ..models.hgrdtw import build_classifier, fit, predict
from ..utils import AssetManager

from ..experiments import run_experiments, print_message, print_progress, print_result, print_line_break, download_assets
from ..experiments.data import emgepn10

def find_optimum_segmentation_threshold(config):
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

def find_optimum_segmentation_thresholds_by_classifier_and_user(
    experiment_id,
    total_experiments,
    dataset_name,
    assets_dir,
    user_ids,
    options,
    threshold_min=10,
    threshold_max=20,
):
    start_ts = datetime.datetime.now()

    classifier_names = options['classifier_names']
    folds = options['cv_folds']

    if 'cv_options' in options.keys():
        cv_options = options['cv_options']
    else:
        cv_options = None

    task = 'Optimizing thresholds'

    print_message(
        'Experiment %d of %d: Optimize HGR segmentation thresholds using %d-fold cross-validation' % (
            experiment_id,
            total_experiments,
            folds,
    ))
    print_message('Classifiers: %s' % classifier_names)
    print_message('Number of subjects: %d' % np.size(user_ids))

    optimum_thresholds = np.zeros((np.size(classifier_names), np.size(user_ids)), dtype=int)

    num_workers = mp.cpu_count()

    def get_progress(classifier_id, config_id=0):
        current = np.size(user_ids) * classifier_id + config_id
        total = np.size(user_ids) * np.size(classifier_names)
        return current / total

    for classifier_id, classifier_name in enumerate(classifier_names):
        print_line_break()
        print_progress(
            task,
            get_progress(classifier_id),
            'running %s-fold cross-validation on classifier %s...' % (
                folds,
                classifier_name,
            )
        )

        user_configs = []

        for user_id in user_ids:
            config = {
                'classifier_name': classifier_name,
                'threshold_min': threshold_min,
                'threshold_max': threshold_max,
                'threshold_direction': 'desc',
                'cv_folds': folds,
                'cv_options': cv_options,
                'feature_set_config': {
                    'user_id': user_id,
                    'ds_name': dataset_name,
                    'ds_type': 'training',
                    'ds_dir': assets_dir,
                    'fs_dir': assets_dir,
                    'stft_window_length': 25,
                    'stft_window_overlap': 10,
                    'stft_nfft': 50,
                    'activity_extra_samples': 25,
                    'activity_min_length': 100,
                },
            }
            
            user_configs.append(config)
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            for i, config, result in zip(
                np.arange(0, np.size(user_configs)),
                user_configs,
                executor.map(
                    find_optimum_segmentation_threshold,
                    user_configs,
                )
            ):
                print_progress(
                    task,
                    get_progress(classifier_id, i+1),
                    'optimized classifier %s for subject %2d of %2d' % (
                        classifier_name,
                        i+1,
                         np.size(user_configs),
                    )
                )
                
                optimum_thresholds[classifier_id,i] = result

    end_ts = datetime.datetime.now()

    output_message = '%s\n%s\n' % (
        'Table 1: Optimum individual segmentation thresholds using 4-fold cross-validation',
        'Lines: classifiers | Columns: subjects'
    )
    for classifier_id, classifier in enumerate(classifier_names):
        output_message = output_message + '\n%03s: %s' % (
            classifier,
            optimum_thresholds[classifier_id],
        )
    
    print_line_break()
    print_message('Finished segmentation threshold optimization')
    print_message('Time elapsed in experiment %d of %d: %s' % (
        experiment_id,
        total_experiments,
        str(end_ts - start_ts),
    ))

    return {
        'data': optimum_thresholds,
        'message': output_message,
    }

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

def assess_hand_gesture_classification(config):
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

def assess_hgr_systems_by_classifier_and_user(
    experiment_id,
    total_experiments,
    dataset_name,
    assets_dir,
    user_ids,
    options,
    experiment_runs=100,
):
    start_ts = datetime.datetime.now()

    classifier_names = options['classifier_names']
    thresholds = options['thresholds']

    task = 'Assessing HGR systems'

    max_workers = mp.cpu_count()
    number_of_users = np.size(user_ids)
    number_of_classifiers = np.size(classifier_names)

    errors = np.zeros(
        (number_of_classifiers,number_of_users,experiment_runs,),
        dtype=np.uint32
    )

    trials = np.zeros(
        (number_of_classifiers,number_of_users,experiment_runs,),
        dtype=np.uint32
    )

    print_message(
        'Experiment %d of %d: Compare the accuracy of HGR systems using %d classifiers ' % (
            experiment_id,
            total_experiments,
            number_of_classifiers,
    ))
    print_message('Classifiers: %s' % classifier_names)
    print_message('Number of subjects: %d' % number_of_users)

    def get_progress(classifier_id, config_id=0):
        current = number_of_users * classifier_id + config_id
        total = number_of_users * number_of_classifiers
        return current / total
    
    for classifier_id, classifier_name in enumerate(classifier_names):
        print_line_break()
        print_progress(
            task,
            get_progress(classifier_id),
            'evaluating classifier %s...' % classifier_name
        )

        user_configs = []

        for i, user_id in enumerate(user_ids):
            optimum_threshold = thresholds[classifier_name][i]
            config = {
                'classifier_name': classifier_name,
                'experiments': experiment_runs,
                'feature_set_config': {
                    'user_id': user_id,
                    'ds_name': dataset_name,
                    'ds_dir': assets_dir,
                    'fs_dir': assets_dir,
                    'stft_window_length': 25,
                    'stft_window_overlap': 10,
                    'stft_nfft': 50,
                    'activity_threshold': optimum_threshold,
                    'activity_extra_samples': 25,
                    'activity_min_length': 100,
                },
            }
            
            user_configs.append(config)

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            for i, config, result in zip(
                np.arange(0, np.size(user_configs)),
                user_configs,
                executor.map(
                    assess_hand_gesture_classification,
                    user_configs,
                )
            ):
                print_progress(
                    task,
                    get_progress(classifier_id, i+1),
                    'evaluated classifier %s for subject %2d of %2d' % (
                        classifier_name,
                        i+1,
                        np.size(user_configs),
                    )
                )
                
                errors[classifier_id,i,:] = result[0]
                trials[classifier_id,i,:] = result[1]

    accuracy = (trials - errors) / trials
    accuracy_mean = accuracy.mean(axis=(1,2))
    accuracy_per_experiment = accuracy.mean(axis=1)
    accuracy_per_user = accuracy.mean(axis=2)

    end_ts = datetime.datetime.now()

    table2 = ''
    for classifier_id, classifier in enumerate(classifier_names):
        table2 = '%s\n%03s: %.1f \u00B1 %.1f' % (
            table2,
            classifier,
            accuracy_mean[classifier_id] * 100,
            accuracy_per_experiment[classifier_id].std(ddof=1) * 100,
        )

    accuracy_by_classifier_output = '%s\n%s\n%s' % (
        'Table 2: Mean accuracy and standard deviation of the HGR systems using different classifiers',
        'Lines: classifiers | Data: mean accuracy and standard deviation (%)',
        table2,
    )
    
    table3 = '    ' + '    '.join(['%12s' % name for name in classifier_names])
    for i, user_id in enumerate(user_ids):
        table3 = table3 + '\n#%02s:' % user_id
        for classifier_id, classifier in enumerate(classifier_names):
            table3 = '{TABLE3}    {MEAN:5.1f} \u00B1 {STD:4.1f}'.format(
                TABLE3=table3,
                MEAN=accuracy_per_user[classifier_id,i] * 100,
                STD=accuracy[classifier_id,i].std(ddof=1) * 100,
            )

    table3 = table3 + '\n\nAVG '
    for classifier_id, classifier in enumerate(classifier_names):
        table3 = '{TABLE3}    {MEAN:5.1f} \u00B1 {STD:4.1f}'.format(
            TABLE3=table3,
            MEAN=accuracy_mean[classifier_id] * 100,
            STD=accuracy_per_user.std(axis=1,ddof=1)[classifier_id] * 100,
        )

    accuracy_by_subject_output = '%s\n%s\n%s' % (
        'Table 3: Mean accuracy and standard deviation by subject for different classifiers',
        'Lines: subjects | Columns: classifiers | Data: mean accuracy and standard deviation (%)',
        table3,
    )

    output_message = '%s\n\n%s' % (
        accuracy_by_classifier_output,
        accuracy_by_subject_output
    )
    
    print_line_break()
    print_message('Finished evaluation of HGR systems')
    print_message('Time elapsed in experiment %d of %d: %s' % (
        experiment_id,
        total_experiments,
        str(end_ts - start_ts),
    ))

    return {
        'data': accuracy,
        'message':  output_message,
    }

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
            find_optimum_segmentation_thresholds_by_classifier_and_user,
            assess_hgr_systems_by_classifier_and_user,
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
            }
        }
    )

if __name__ == '__main__':
    main()
