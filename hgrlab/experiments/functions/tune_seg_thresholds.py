import datetime
import numpy as np
import concurrent.futures
import multiprocessing as mp

from ...experiments import print_message, print_result, print_progress, print_line_break

def run(
    dataset_name,
    assets_dir,
    user_ids,
    options,
    tune_segmentation_threshold,
    threshold_min=10,
    threshold_max=20,
):
    start_ts = datetime.datetime.now()

    classifier_names = options['classifier_names']
    folds = options['cv_folds']

    feature_window_length = options['feature_window_length']
    feature_overlap_length = options['feature_overlap_length']

    dtw_impl = options['dtw_impl']

    if 'cv_options' in options.keys():
        cv_options = options['cv_options']
    else:
        cv_options = None

    if 'classifier_options' in options.keys():
        classifier_options = options['classifier_options']
    else:
        classifier_options = None

    task = 'Optimizing segmentation thresholds'

    print_line_break()
    print_message(
        'Optimize HGR segmentation thresholds using %d-fold cross-validation' % (
            folds,
    ))
    print_message('Classifiers: %s' % classifier_names)
    print_message('Number of subjects: %d' % np.size(user_ids))

    optimum_thresholds = np.zeros((np.size(classifier_names), np.size(user_ids)), dtype=int)
    threshold_errors = np.zeros((np.size(classifier_names), np.size(user_ids)), dtype=int)
    threshold_predictions = np.zeros((np.size(classifier_names), np.size(user_ids)), dtype=int)

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
            'running %s-fold CV on classifier %s...' % (
                folds,
                classifier_name,
            )
        )

        user_configs = []

        if classifier_options and classifier_name in classifier_options:
            current_classifier_options = classifier_options[classifier_name]
        else:
            current_classifier_options = None

        for user_id in user_ids:
            config = {
                'threshold_min': threshold_min,
                'threshold_max': threshold_max,
                'threshold_direction': 'desc',
                'cv_folds': folds,
                'cv_options': cv_options,
                'classifier_name': classifier_name,
                'classifier_options': current_classifier_options,
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
                    'feature_window_length': feature_window_length,
                    'feature_overlap_length': feature_overlap_length,
                    'dtw_impl': dtw_impl,
                },
            }
            
            user_configs.append(config)
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            for i, config, result in zip(
                np.arange(0, np.size(user_configs)),
                user_configs,
                executor.map(
                    tune_segmentation_threshold,
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
                
                optimum_thresholds[classifier_id,i] = result['threshold']
                threshold_errors[classifier_id,i] = result['errors']
                threshold_predictions[classifier_id,i] = result['predictions']

    end_ts = datetime.datetime.now()

    output_message = '%s\n%s\n' % (
        'Optimum individual segmentation thresholds using 4-fold cross-validation',
        'Lines: classifiers | Columns: subjects'
    )
    for classifier_id, classifier in enumerate(classifier_names):
        output_message = output_message + '\n%03s: %s' % (
            classifier,
            optimum_thresholds[classifier_id],
        )

    print_line_break()
    print_message('Validation accuracy:')

    for classifier_id, classifier in enumerate(classifier_names):
        errors = threshold_errors[classifier_id].sum()
        predictions = threshold_predictions[classifier_id].sum()

        print_result('%03s_%d-fold_CV_acc = %.1f (%d/%d)' % (
            classifier,
            folds,
            (1 - errors / predictions) * 100,
            errors,
            predictions,
        ))

    print_line_break()
    print_message(
        'Finished segmentation threshold optimization (time elapsed: {INTERVAL})'.format(
            INTERVAL=str(end_ts - start_ts),
        )
    )

    return {
        'data': optimum_thresholds,
        'message': output_message,
    }
