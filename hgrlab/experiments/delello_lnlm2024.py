import os

import numpy as np
import multiprocessing as mp
import concurrent.futures
import datetime

from ..models.hgrdtw import k_fold_cost, FeatureSet
from ..models.hgrdtw import build_classifier, fit, predict
from ..utils import AssetManager

from ..experiments import run_experiments, print_message, print_progress, print_result, print_line_break

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
            'filename': 'subject1_training.h5',
        },
        'subject2_training': {
            'remote_id': '1-Gjn9KKoFTVH7_O-9L4wOyl2UzX3mIj6',
            'filename': 'subject2_training.h5',
        },
        'subject3_training': {
            'remote_id': '1-r5A0aL4hI9rMsLLGO5O5rgjriTrLMUM',
            'filename': 'subject3_training.h5',
        },
        'subject4_training': {
            'remote_id': '1-8BmKmR4FTVLc8cggZkouCO7BIRHqFuK',
            'filename': 'subject4_training.h5',
        },
        'subject5_training': {
            'remote_id': '1-f0m6uqj3AFMbzA5769TBaEFZDfh6WGa',
            'filename': 'subject5_training.h5',
        },
        'subject6_training': {
            'remote_id': '10BF3MuHphVVq1Mpn-YJ13--VvyQR5HuF',
            'filename': 'subject6_training.h5',
        },
        'subject7_training': {
            'remote_id': '1--vbiMvJL9pvUXOmAr9efVM9iPq-UYog',
            'filename': 'subject7_training.h5',
        },
        'subject8_training': {
            'remote_id': '1-tQbeFzUFqkoun5yXWBvtgYrRycPzuTd',
            'filename': 'subject8_training.h5',
        },
        'subject9_training': {
            'remote_id': '103OiJYh-b6gWv5Yew3JTwFo2S7LgbmE7',
            'filename': 'subject9_training.h5',
        },
        'subject10_training': {
            'remote_id': '1-3aMg8xDAVBknSLWnNbNtX8dHiNjZofe',
            'filename': 'subject10_training.h5',
        },
        'subject1_test_features_1': {
            'remote_id': '1rLtTvjUC-Ms6HqZJsstIQOB3ZR4hcoi8',
            'filename': 'emgepn30_features_subject1_test_1b57d5b091d4ad3a12ef5ecd749dc5e7.h5',
        },
        'subject1_test_features_2': {
            'remote_id': '1rtASIrqwxy9vqmuEWBbJRCor3tPYUWKl',
            'filename': 'emgepn30_features_subject1_test_275faa72d1fde5ff75d575839e2fc4b4.h5',
        },
        'subject2_test_features_1': {
            'remote_id': '1rDxrwPqV8JrGkAGThfSPMW4F3S5lerWF',
            'filename': 'emgepn30_features_subject2_test_bc6ff8cbac25292050958fd345ef3fa8.h5',
        },
        'subject2_test_features_2': {
            'remote_id': '1sXdhyWOUNqtbXgX6_Dw4XaaGRhbQ6kRW',
            'filename': 'emgepn30_features_subject2_test_c450226bdfd56d2a447763c8b143e32e.h5',
        },
        'subject3_test_features_1': {
            'remote_id': '1sPIk5jzGWuop7F6tBMW2a94mMoeapZB7',
            'filename': 'emgepn30_features_subject3_test_622e160136988940243023c1c2eca08e.h5',
        },
        'subject3_test_features_2': {
            'remote_id': '1rObfE33ZvrojGMGyLd4EdTH8ykLltYys',
            'filename': 'emgepn30_features_subject3_test_7602c145973b875375978741675d200f.h5',
        },
        'subject3_test_features_3': {
            'remote_id': '1rWPPYvIxR0_RBacS8buw-EIt8N3VyyfE',
            'filename': 'emgepn30_features_subject3_test_bdced377ed1b9119ede3ac699b24c7d5.h5',
        },
        'subject3_test_features_4': {
            'remote_id': '1rtUwTRMUz1X7L_cUotdAJq6IkCgUyQX6',
            'filename': 'emgepn30_features_subject3_test_d6a49b2e85d2f09ca6d215806a985432.h5',
        },
        'subject4_test_features_1': {
            'remote_id': '1tVkftGuh6oTFBMS8G-QJ7aA8mqJh0QTQ',
            'filename': 'emgepn30_features_subject4_test_9f6c2bc1afc7c34f90e412976c1299d1.h5',
        },
        'subject4_test_features_2': {
            'remote_id': '1sKiuZSxynHbD5lQJT3R2-XEMoazhkFTk',
            'filename': 'emgepn30_features_subject4_test_dc5126d46ccf581299e73d9286364873.h5',
        },
        'subject5_test_features_1': {
            'remote_id': '1rvq3st8Qm_AtleCF9HuCm31LUso79axh',
            'filename': 'emgepn30_features_subject5_test_052ead8353b5136d5b4b1ae407437ee4.h5',
        },
        'subject5_test_features_2': {
            'remote_id': '1t2EqKbequhVMXkgrPv2X-SF6IuS1jFcr',
            'filename': 'emgepn30_features_subject5_test_96157d0333c8b5523942efd86d1627ef.h5',
        },
        'subject5_test_features_3': {
            'remote_id': '1rF6afXyn-eRWer1LiH00krg0o1Soj2EO',
            'filename': 'emgepn30_features_subject5_test_b11c9c2bed8ec31d7cbf9e76436466f5.h5',
        },
        'subject5_test_features_4': {
            'remote_id': '1sz0bk1ueV9SzRh9CD7nYNilNwLKBHuAS',
            'filename': 'emgepn30_features_subject5_test_d75f0326c7b26ee2cc9bcb97694373e7.h5',
        },
        'subject6_test_features_1': {
            'remote_id': '1sJFQiEJ4mK7L5XJlO2UkRGBu0eC6B6D-',
            'filename': 'emgepn30_features_subject6_test_7da80703509bc75a0dd01d811427d9c2.h5',
        },
        'subject6_test_features_2': {
            'remote_id': '1rEqgCLf6aev-XNLfz9RCkAxB2TDPmns4',
            'filename': 'emgepn30_features_subject6_test_69d0e5db3aa966f6b368db5ade9d4774.h5',
        },
        'subject6_test_features_3': {
            'remote_id': '1s0286jq8AgP-_IwHRcKJdJtFZOC_nYdL',
            'filename': 'emgepn30_features_subject6_test_4304c3177801b2c6f288a2e92f90c2aa.h5',
        },
        'subject7_test_features_1': {
            'remote_id': '1ro0uJ0xObPdbJjnBSMANtTPKJyzXtbgp',
            'filename': 'emgepn30_features_subject7_test_51dd2cd2ba70dcc68288bbce53f56670.h5',
        },
        'subject7_test_features_2': {
            'remote_id': '1sI395UW55qEQiEz1KvGVPCa8DLzIR7RI',
            'filename': 'emgepn30_features_subject7_test_d1bcff6dca1f9f4753da18961ce384a3.h5',
        },
        'subject8_test_features_1': {
            'remote_id': '1tsizP4gw0upYnKqOmX0-eiYN3Mmigl74',
            'filename': 'emgepn30_features_subject8_test_78c926c1455722501ad0bc662a2cff11.h5',
        },
        'subject9_test_features_1': {
            'remote_id': '1roMDn9gHZbhzBWhvPdIwlrdqDunjiZqd',
            'filename': 'emgepn30_features_subject9_test_b627bacddbdd8ce47b7a289fa05b2a03.h5',
        },
        'subject10_test_features_1': {
            'remote_id': '1sJHSYMa3zW4Eao9tbWbI2tlWSCIX0aZj',
            'filename': 'emgepn30_features_subject10_test_55fb5ad833c5dab92e8436482a6460bb.h5',
        },
        'subject10_test_features_2': {
            'remote_id': '1tcipgt0GXLWDyTfveeO40VzCqc9mZuhC',
            'filename': 'emgepn30_features_subject10_test_637970283bcf00faf5b7c17542794b4d.h5',
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

    run_experiments(
        title,
        dataset_name='emgepn30',
        assets_dir=AssetManager.get_base_dir(),
        user_ids=np.arange(1, 11),
        setup=download_assets,
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
