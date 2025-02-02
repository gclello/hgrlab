import datetime
import numpy as np
import concurrent.futures
import multiprocessing as mp

from ...experiments import print_message, print_progress, print_line_break

def run(
    experiment_id,
    total_experiments,
    dataset_name,
    assets_dir,
    user_ids,
    options,
    eval_hgr_system,
    experiment_runs=100,
):
    start_ts = datetime.datetime.now()

    classifier_names = options['classifier_names']
    thresholds = options['thresholds']
    feature_window_length = options['feature_window_length']
    feature_overlap_length = options['feature_overlap_length']

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
                    'feature_window_length': feature_window_length,
                    'feature_overlap_length': feature_overlap_length,
                },
            }
            
            user_configs.append(config)

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            for i, config, result in zip(
                np.arange(0, np.size(user_configs)),
                user_configs,
                executor.map(
                    eval_hgr_system,
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

    acc_by_classifier_output = ''
    for classifier_id, classifier in enumerate(classifier_names):
        acc_by_classifier_output = '%s\n%03s: %.1f \u00B1 %.1f' % (
            acc_by_classifier_output,
            classifier,
            accuracy_mean[classifier_id] * 100,
            accuracy_per_experiment[classifier_id].std(ddof=1) * 100,
        )

    acc_by_classifier_output = '%s\n%s\n%s' % (
        'Mean accuracy and standard deviation of the HGR systems using different classifiers',
        'Lines: classifiers | Data: mean accuracy and standard deviation (%)',
        acc_by_classifier_output,
    )
    
    acc_by_subject_output = '    ' + '    '.join(
        ['%12s' % name for name in classifier_names]
    )

    for i, user_id in enumerate(user_ids):
        acc_by_subject_output = acc_by_subject_output + '\n#%02s:' % user_id
        for classifier_id, classifier in enumerate(classifier_names):
            acc_by_subject_output = '{OUTPUT}    {MEAN:5.1f} \u00B1 {STD:4.1f}'.format(
                OUTPUT=acc_by_subject_output,
                MEAN=accuracy_per_user[classifier_id,i] * 100,
                STD=accuracy[classifier_id,i].std(ddof=1) * 100,
            )

    acc_by_subject_output = acc_by_subject_output + '\n\nAVG '
    for classifier_id, classifier in enumerate(classifier_names):
        acc_by_subject_output = '{OUTPUT}    {MEAN:5.1f} \u00B1 {STD:4.1f}'.format(
            OUTPUT=acc_by_subject_output,
            MEAN=accuracy_mean[classifier_id] * 100,
            STD=accuracy_per_user.std(axis=1,ddof=1)[classifier_id] * 100,
        )

    acc_by_subject_output = '%s\n%s\n%s' % (
        'Mean accuracy and standard deviation by subject for different classifiers',
        'Lines: subjects | Columns: classifiers | Data: mean accuracy and standard deviation (%)',
        acc_by_subject_output,
    )

    output_message = '%s\n\n%s' % (
        acc_by_classifier_output,
        acc_by_subject_output,
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
        'message': output_message,
    }
