import os

import numpy as np
import multiprocessing as mp
import concurrent.futures
import datetime

from ..models.hgrdtw import k_fold_classification_cost
from ..models.hgrdtw import build_classifier, fit, predict
from ..utils import load_pickle
from ..utils import AssetManager

from ..experiments import run_experiments, print_message, print_title, print_progress, print_result, print_line_break

def find_optimum_segmentation_threshold(config):
    threshold_min = config['threshold_min']
    threshold_max = config['threshold_max']
    threshold_direction = config['threshold_direction']

    if threshold_direction == 'desc':
        thresholds = np.flip(np.arange(threshold_min, threshold_max+1))
    else:
        thresholds = np.arange(threshold_min, threshold_max+1)

    HUGE_ERROR = 1000
    thresholds_errors = np.full(np.size(thresholds), HUGE_ERROR)

    for threshold_id, threshold in enumerate(thresholds):
        config['activity_threshold'] = threshold

        errors, predictions = k_fold_classification_cost(config)

        thresholds_errors[threshold_id] = errors

        if errors == 0:
            break

    return thresholds[np.argmax(thresholds_errors)]

def find_optimum_segmentation_thresholds_by_classifier_and_user(
    experiment_id,
    total_experiments,
    assets_dir,
    classifier_names,
    user_ids,
    threshold_min=10,
    threshold_max=20,
):
    start_ts = datetime.datetime.now()

    task = 'Optimizing thresholds'

    folds = 4
    val_size_per_class = 2

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
                'database_path': assets_dir,
                'database_type': 'training',
                'classifier_name': classifier_name,
                'user_id': user_id,
                'threshold_min': threshold_min,
                'threshold_max': threshold_max,
                'threshold_direction': 'desc',
                'cross_validation_folds': folds,
                'cross_validation_val_size_per_class': val_size_per_class,
                'stft_window_length': 25,
                'stft_window_overlap': 10,
                'stft_nfft': 50,
                'activity_extra_samples': 25,
                'activity_min_length': 100,
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

    result_message = '%s\n%s\n' % (
        'Table 1: Optimum individual segmentation thresholds using 4-fold cross-validation',
        'Lines: classifiers | Columns: subjects'
    )
    for classifier_id, classifier in enumerate(classifier_names):
        result_message = result_message + '\n%03s: %s' % (
            classifier,
            optimum_thresholds[classifier_id],
        )
    
    print_line_break()
    print_result(result_message)
    print_line_break()
    print_message('Finished segmentation threshold optimization')
    print_message('Time elapsed in experiment %d of %d: %s' % (
        experiment_id,
        total_experiments,
        str(end_ts - start_ts),
    ))

    return optimum_thresholds

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
    user_id = config['user_id']
    classifier_name = config['classifier_name']
    training_file = config['training_file']
    test_file = config['test_file']

    training_data = load_pickle(training_file)
    test_data = load_pickle(test_file)

    X_train = training_data[user_id]['features']
    y_train = training_data[user_id]['labels']
    X_test = test_data[user_id]['features']
    y_test = test_data[user_id]['labels']
    test_activity_indices = test_data[user_id]['activity_indexes']

    test_trials = X_test.shape[0]
    test_window_length = X_test.shape[2]

    errors = np.zeros((experiments),dtype=np.uint32)
    trials = np.zeros((experiments),dtype=np.uint32)

    for experiment in np.arange(0, experiments):
        model = build_classifier(classifier_name)
        model.fit(X_train, y_train)

        prediction = np.full((test_trials,), 'relax', dtype='U14')
        
        for trial_id, X_test_windows in enumerate(X_test):
            test_window_predictions = model.predict(X_test_windows)

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
    assets_dir,
    classifier_names,
    user_ids,
    experiment_runs=100,
):
    start_ts = datetime.datetime.now()

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

        training_file = os.path.join(
            assets_dir,
            'training_features_%s.pickle' % classifier_name,
        )

        test_file = os.path.join(
            assets_dir,
            'test_features_m500_s10_%s.pickle' % classifier_name,
        )

        for user_id in user_ids:
            config = {
                'training_file': training_file,
                'test_file': test_file,
                'classifier_name': classifier_name,
                'user_id': user_id,
                'experiments': experiment_runs,
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
    
    print_line_break()
    print_result(accuracy_by_classifier_output)
    print_line_break()
    print_result(accuracy_by_subject_output)
    print_line_break()
    print_message('Finished evaluation of HGR systems')
    print_message('Time elapsed in experiment %d of %d: %s' % (
        experiment_id,
        total_experiments,
        str(end_ts - start_ts),
    ))

def download_assets():
    start_ts = datetime.datetime.now()
    
    task = 'Download HGR dataset'

    asset_manager = AssetManager()

    assets = {
        'subject1_training': {
            'remote_id': '1-F1p0N5x2C3fey9GupDYlmHob0l_ZIMg',
            'filename': 'subject1_training.h5'
        },
        'subject2_training': {
            'remote_id': '1-Gjn9KKoFTVH7_O-9L4wOyl2UzX3mIj6',
            'filename': 'subject2_training.h5'
        },
        'subject3_training': {
            'remote_id': '1-r5A0aL4hI9rMsLLGO5O5rgjriTrLMUM',
            'filename': 'subject3_training.h5'
        },
        'subject4_training': {
            'remote_id': '1-8BmKmR4FTVLc8cggZkouCO7BIRHqFuK',
            'filename': 'subject4_training.h5'
        },
        'subject5_training': {
            'remote_id': '1-f0m6uqj3AFMbzA5769TBaEFZDfh6WGa',
            'filename': 'subject5_training.h5'
        },
        'subject6_training': {
            'remote_id': '10BF3MuHphVVq1Mpn-YJ13--VvyQR5HuF',
            'filename': 'subject6_training.h5'
        },
        'subject7_training': {
            'remote_id': '1--vbiMvJL9pvUXOmAr9efVM9iPq-UYog',
            'filename': 'subject7_training.h5'
        },
        'subject8_training': {
            'remote_id': '1-tQbeFzUFqkoun5yXWBvtgYrRycPzuTd',
            'filename': 'subject8_training.h5'
        },
        'subject9_training': {
            'remote_id': '103OiJYh-b6gWv5Yew3JTwFo2S7LgbmE7',
            'filename': 'subject9_training.h5'
        },
        'subject10_training': {
            'remote_id': '1-3aMg8xDAVBknSLWnNbNtX8dHiNjZofe',
            'filename': 'subject10_training.h5'
        },
        'training_features_svm': {
            'remote_id': '10UBEz8HBpkAwKRF6xlgG_K7LGCFM7qlW',
            'filename': 'training_features_svm.pickle'
        },
        'test_features_svm': {
            'remote_id': '13pfALxZ8u_dzi_b7c2RBmlQeNXLBRgVt',
            'filename': 'test_features_m500_s10_svm.pickle'
        },
        'training_features_lr': {
            'remote_id': '1a1wgDWTYZsvAfhs-Ltu-6wTOo4kGvTiL',
            'filename': 'training_features_lr.pickle'
        },
        'test_features_lr': {
            'remote_id': '1j4J3Yub1ksg6Y1hc9nggdKgIHuaIN300',
            'filename': 'test_features_m500_s10_lr.pickle'
        },
        'training_features_lda': {
            'remote_id': '1hKesrsPOf0_kCO0lhhEjJhipIupDXFqe',
            'filename': 'training_features_lda.pickle'
        },
        'test_features_lda': {
            'remote_id': '1u9b29jWvO0ZNFv-LeHj-aH5GtP5fcmFv',
            'filename': 'test_features_m500_s10_lda.pickle'
        },
        'training_features_knn': {
            'remote_id': '1n2hwI-9voM8ImDBTZ1hPcXd1PaMaJjAw',
            'filename': 'training_features_knn.pickle'
        },
        'test_features_knn': {
            'remote_id': '1ghknJcGayi1EOEUvFNx1jbHUz5XYzvuW',
            'filename': 'test_features_m500_s10_knn.pickle'
        },
        'training_features_dt': {
            'remote_id': '1582s-IC2ThcACjJ7WdcuNODyd_0UwTsG',
            'filename': 'training_features_dt.pickle'
        },
        'test_features_dt': {
            'remote_id': '12PbHAr_Jo_u-HUibMj2lDCMalta9gZvU',
            'filename': 'test_features_m500_s10_dt.pickle'
        },
    }

    total_files =  len(assets.keys())

    print_progress(
        task,
        progress=0.0,
        status='downloading %d files...' % total_files,
    )

    for i, key in enumerate(assets.keys()):
        asset_manager.add_remote_asset(
            key,
            assets[key]['remote_id'],
            assets[key]['filename'],
        )

        cached = asset_manager.download_asset(key)

        if cached:
            message = 'found file %2d of %2d in local cache (%s)'
        else:
            message = 'downloaded file %2d of %2d (%s)' 

        print_progress(
            task,
            progress=(i+1)/total_files,
            status=message % (
                i+1,
                total_files,
                assets[key]['filename'],
            ),
        )
    
    end_ts = datetime.datetime.now()
    print_line_break()
    print_message('Finished downloading files')
    print_message('Time elapsed downloading files: %s' % str(end_ts - start_ts))

    return asset_manager

def main():
    authors = 'Guilherme C. De Lello, Gabriel S. Chaves, Juliano F. Caldeira, and Markus V.S. Lima'
    title = 'HGR experiments conducted by %s on March 2024' % authors

    classifiers_names = [
        'svm',
        'lr',
        'lda',
        'knn',
        'dt',
    ]

    user_ids = np.arange(1, 11)

    experiments = [
        find_optimum_segmentation_thresholds_by_classifier_and_user,
        assess_hgr_systems_by_classifier_and_user,
    ]

    run_experiments(
        title,
        setup=download_assets,
        experiments=experiments,
        assets_dir=AssetManager.get_base_dir(),
        classifier_names=classifiers_names,
        user_ids=user_ids,
    )

if __name__ == '__main__':
    main()
