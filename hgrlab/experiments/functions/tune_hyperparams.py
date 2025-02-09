import datetime
import numpy as np
import concurrent.futures
import multiprocessing as mp

from ...experiments import print_message, print_result, print_progress, print_line_break

def run(
    dataset_name,
    ds_dir,
    fs_dir,
    user_ids,
    options,
    cost_function,
):
    start_ts = datetime.datetime.now()

    thresholds = options['thresholds']
    classifier_name = options['classifier_name']
    classifier_options_list = options['classifier_options_list']

    folds = options['cv_folds']

    if 'cv_options' in options.keys():
        cv_options = options['cv_options']
    else:
        cv_options = None

    feature_window_length = options['feature_window_length']
    feature_overlap_length = options['feature_overlap_length']

    dtw_impl = options['dtw_impl']

    task = 'Optimizing hyperparameters'

    print_line_break()
    print_message(
        'Optimize hyperparameters using %d-fold cross-validation' % (
            folds,
    ))
    print_message('Classifier: %s' % classifier_name)
    print_message('Number of subjects: %d' % np.size(user_ids))
    print_message('Number of hyperparameter configurations: %d' % np.size(classifier_options_list))

    options_errors = np.zeros((np.size(user_ids), np.size(classifier_options_list)), dtype=int)
    options_predictions = np.zeros((np.size(user_ids), np.size(classifier_options_list)), dtype=int)
    options_best_indices = np.zeros((np.size(user_ids)), dtype=int)
    best_options = {}

    num_workers = mp.cpu_count()

    def get_progress(user_index, config_id=0):
        current = np.size(classifier_options_list) * user_index + config_id
        total = np.size(user_ids) * np.size(classifier_options_list)
        return current / total

    for i, user_id in enumerate(user_ids):
        print_line_break()
        print_progress(
            task,
            get_progress(i),
            'running %s-fold CV on classifier %s for subject %2d...' % (
                folds,
                classifier_name,
                user_id,
            )
        )
        
        configs = []
        
        for classifier_options in classifier_options_list:
            config = {
                'cv_folds': folds,
                'cv_options': cv_options,
                'classifier_name': classifier_name,
                'classifier_options': classifier_options,
                'feature_set_config': {
                    'user_id': user_id,
                    'ds_name': dataset_name,
                    'ds_type': 'training',
                    'ds_dir': ds_dir,
                    'fs_dir': fs_dir,
                    'stft_window_length': 25,
                    'stft_window_overlap': 10,
                    'stft_nfft': 50,
                    'activity_threshold': thresholds[classifier_name][user_id-1],
                    'activity_extra_samples': 25,
                    'activity_min_length': 100,
                    'feature_window_length': feature_window_length,
                    'feature_overlap_length': feature_overlap_length,
                    'dtw_impl': dtw_impl,
                },
            }
            
            configs.append(config)
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            for j, config, result in zip(
                np.arange(0, np.size(configs)),
                configs,
                executor.map(
                    cost_function,
                    configs,
                )
            ):
                print_progress(
                    task,
                    get_progress(i, j+1),
                    'tuned classifier %s for subject %2d (%2d of %2d)' % (
                        classifier_name,
                        user_id,
                        j+1,
                        np.size(configs),
                    )
                )
                
                options_errors[i,j] = result['fold_errors'].sum()
                options_predictions[i,j] = result['fold_predictions'].sum()
                if options_errors[i,j] == 0:
                    break

    end_ts = datetime.datetime.now()

    table_val_acc = '{TITLE}\n{CAPTION}\n{HEADER}'.format(
        TITLE='Hyperparameters tuning using %s-fold CV for classifier %s' % (folds,classifier_name),
        CAPTION='Lines: subjects',
        HEADER='{IDENT}{COLUMNS}'.format(
            IDENT='   ',
            COLUMNS='   '.join(['%12s' % name for name in [
                'Val. acc.',
                'Errors / Predictions',
            ]])
        )
    )

    table_best_options = '{TITLE}\n{CAPTION}\n'.format(
        TITLE='Optimal hyperparameters for classifier %s' % classifier_name,
        CAPTION='Lines: subjects | Data: best hyperparameters',
    )

    for i, user_id in enumerate(user_ids):
        options_best_indices[i] = np.argmin(options_errors[i])
        best_options[user_id] = classifier_options_list[options_best_indices[i]]
        errors = options_errors[i][options_best_indices[i]]
        predictions = options_predictions[i][options_best_indices[i]]
        table_val_acc = '{PREVIOUS}\n#{SUBJ:2s}    {ACC:5.1f}%          {ERR} / {PRED}'.format(
            PREVIOUS=table_val_acc,
            SUBJ=str(user_id),
            ACC=(1 - errors / predictions) * 100,
            ERR=errors,
            PRED=predictions,
            OPTIONS=best_options[user_id]
        )
        table_best_options = '{PREVIOUS}\n#{SUBJ:2s}    {OPTIONS}'.format(
            PREVIOUS=table_best_options,
            SUBJ=str(user_id),
            OPTIONS=best_options[user_id]
        )
    print_line_break()
    print_message(
        'Finished hyperparameters optimization (time elapsed: {INTERVAL})'.format(
            INTERVAL=str(end_ts - start_ts),
        )
    )

    return {
        'data': {
            'best_options': best_options,
            'errors': options_predictions,
            'predictions': options_predictions,
        },
        'message': '{TABLE_ACC}\n\n{TABLE_HYPERPARAMS}'.format(
            TABLE_ACC=table_val_acc,
            TABLE_HYPERPARAMS=table_best_options,
        )
    }
