import numpy as np
import multiprocessing as mp
import concurrent.futures
import datetime

from ..models.hgrdtw import k_fold_classification_cost
from ..utils.asset_manager import AssetManager

def get_formatted_time():
    now = datetime.datetime.now()
    return '{YEAR:04d}-{MONTH:02d}-{DAY:02d} {HOUR:02d}:{MINUTE:02d}:{SECOND:02d}'.format(
        YEAR=now.year,
        MONTH=now.month,
        DAY=now.day,
        HOUR=now.hour,
        MINUTE=now.minute,
        SECOND=now.second,
    )

def print_message(message):
    print('{DATE} {MESSAGE}'.format(
        DATE=get_formatted_time(),
        MESSAGE=message,
    ))

def print_line_break():
    print()

def print_progress(task, progress, status):
    print('{DATE} [{PROGRESS:5.1f}%] {TASK}: {STATUS}'.format(
        DATE=get_formatted_time(),
        TASK=task,
        PROGRESS=progress*100,
        STATUS=status,
    ))

def find_optimum_segmentation_threshold(config):
    classifier_name = config['classifier_name']
    folds = config['folds']
    val_size_per_class = config['val_size_per_class']
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

        errors = k_fold_classification_cost(
            config,
            classifier_name,
            folds,
            val_size_per_class,
        )

        thresholds_errors[threshold_id] = errors

        if errors == 0:
            break

    return thresholds[np.argmax(thresholds_errors)]

def find_optimum_segmentation_thresholds_by_classifier_and_user(
    experiment_id,
    total_experiments,
    dataset_path,
    classifier_names,
    user_ids,
    threshold_min,
    threshold_max,
):
    start_ts = datetime.datetime.now()

    task = 'Optimizing thresholds'

    folds = 4
    val_size_per_class = 2
    window_length = 500
    window_stride = 10

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
            'running %s-fold cross-validation on classifier %s' % (
                folds,
                classifier_name,
            )
        )

        user_configs = []

        for user_id in user_ids:
            config = {
                'database_path': dataset_path,
                'database_type': 'training',
                'classifier_name': classifier_name,
                'user_id': user_id,
                'threshold_min': threshold_min,
                'threshold_max': threshold_max,
                'threshold_direction': 'desc',
                'folds': folds,
                'val_size_per_class': val_size_per_class,
                'feature_window_length': window_length,
                'feature_overlap_length': window_length - window_stride,
                'stft_window_length': 25,
                'stft_window_overlap': 10,
                'stft_nfft': 50,
                'activity_extra_samples': 25,
                'activity_min_length': 100,
            }
            
            user_configs.append(config)
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            for i, config, result in zip(
                np.arange(0, len(user_configs)),
                user_configs,
                executor.map(
                    find_optimum_segmentation_threshold,
                    user_configs,
                )
            ):
                print_progress(
                    task,
                    get_progress(classifier_id, i+1),
                    'optimized classifier %s for user %2d of %2d' % (
                        classifier_name,
                        i+1,
                        len(user_configs),
                    )
                )
                
                optimum_thresholds[classifier_id,i] = result

    end_ts = datetime.datetime.now()

    print_line_break()
    print_message('Finished segmentation threshold optimization')

    result_message = 'Optimum segmentation thresholds by classifier (lines) and user (columns):\n'
    for classifier_id, classifier in enumerate(classifier_names):
        result_message = result_message + '\n%03s: %s' % (
            classifier,
            optimum_thresholds[classifier_id],
        )
    
    print_line_break()
    print_message(result_message)
    print_line_break()
    print_message('Time elapsed in experiment %d of %d: %s' % (
        experiment_id,
        total_experiments,
        str(end_ts - start_ts),
    ))

    return optimum_thresholds

def download_assets():
    start_ts = datetime.datetime.now()
    
    task = 'Download HGR dataset'

    assets = AssetManager()

    semg_training_files = {
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
    }

    total_files = len(semg_training_files.keys())

    print_progress(
        task,
        progress=0.0,
        status='downloading %d files' % total_files,
    )

    for i, key in enumerate(semg_training_files.keys()):
        assets.add_remote_asset(
            key,
            semg_training_files[key]['remote_id'],
            semg_training_files[key]['filename'],
        )

        cached = assets.download_asset(key)

        if cached:
            message = 'found file %d of %d in local cache (%s)'
        else:
            message = 'downloaded file %d of %d (%s)' 

        print_progress(
            task,
            progress=(i+1)/total_files,
            status=message % (
                i+1,
                total_files,
                semg_training_files[key]['filename'],
            ),
        )
    
    end_ts = datetime.datetime.now()
    print_line_break()
    print_message('Finished downloading files')
    print_line_break()
    print_message('Time elapsed downloading files: %s' % str(end_ts - start_ts))

    return assets

def main():
    start_ts = datetime.datetime.now()

    print_line_break()
    print_message('HGR experiments conducted by De Lello et al., March 2024')
    print_line_break()
    download_assets()

    classifiers_names = [
        'svm',
        'lr',
        'lda',
        'knn',
        'dt',
    ]

    user_ids = np.arange(1, 11)

    print_line_break()
    find_optimum_segmentation_thresholds_by_classifier_and_user(
        experiment_id=1,
        total_experiments=1,
        dataset_path=AssetManager.get_base_dir(),
        classifier_names=classifiers_names,
        user_ids=user_ids,
        threshold_min=10,
        threshold_max=20,
    )

    end_ts = datetime.datetime.now()
    print_line_break()
    print_message('Finished all experiments')
    print_line_break()
    print_message('Total time elapsed: %s' % str(end_ts - start_ts))
    print_line_break()

if __name__ == '__main__':
    main()
