import datetime

import numpy as np
import scipy
import scipy.signal

from ...emgts import EmgTrialSet
from .segmentation import get_activity_indices
from .segmentation import get_activity_indices_from_trial_set
from .segmentation import get_activity_indices_from_trial_set_windows
from .preprocessing import preprocess

def fastdtw_distance(series1, series2):
    import fastdtw

    distance, _ = fastdtw.fastdtw(
        series1,
        series2,
        radius=100,
        dist=2,
    )

    return distance

def dtai_distance(series1, series2):
    import dtaidistance

    return dtaidistance.dtw.distance_fast(
        np.array(series1, dtype=np.double),
        np.array(series2, dtype=np.double),
        use_c=True,
        use_ndim=True,
    )

def dtw_distance(series1, series2, dtw_impl):
    if dtw_impl == 'fastdtw':
        return fastdtw_distance(series1, series2)
    
    if dtw_impl == 'dtaidistance':
        return dtai_distance(series1, series2)
    
    raise Exception('Invalid DTW distance implementation')

def dtw_from_all_trials(trials, trials_indices, dtw_impl):
    '''Compute DTW distances among all trials'''

    number_of_trials = len(trials)
    dtw_matrix = np.zeros((number_of_trials, number_of_trials))
    
    for i in np.arange(0, trials.shape[0]):
        for j in np.arange(0, trials.shape[0]):
            if j > i:
                distance = dtw_distance(
                    trials[i][trials_indices[i][0]:trials_indices[i][1]],
                    trials[j][trials_indices[j][0]:trials_indices[j][1]],
                    dtw_impl,
                )
                
                dtw_matrix[i, j] = dtw_matrix[j, i] = distance
                
    return dtw_matrix

def get_class_center_trial_id(trials, trials_indices, trials_labels, class_label, dtw_impl):
    '''Get the trial_id that is the nearest among all other trial_ids from the same class'''

    dtw_matrix = dtw_from_all_trials(
        trials[trials_labels == class_label],
        trials_indices[trials_labels == class_label],
        dtw_impl,
    )

    min_cost_index = dtw_matrix.sum(axis=0).argmin()
    return np.argwhere(trials_labels == class_label)[min_cost_index][0]

def dtw_from_specific_trial_ids(trials, trials_indices, trial_ids, dtw_impl):
    '''Compute DTW distances between each trial and specific trials'''

    number_of_trials = len(trials)
    number_of_trial_ids = len(trial_ids)
    dtw_matrix = np.zeros((number_of_trials, number_of_trial_ids))
    
    for i, trial_data in enumerate(trials):
        for j, trial_id in enumerate(trial_ids):
            distance = dtw_distance(
                trial_data[trials_indices[i][0]:trials_indices[i][1]],
                trials[trial_id][trials_indices[trial_id][0]:trials_indices[trial_id][1]],
                dtw_impl,
            )
                  
            dtw_matrix[i, j] = distance
                
    return dtw_matrix

def dtw_between_two_series(
    series1,
    series1_indices,
    series1_ids,
    series2,
    series2_indices,
    series2_ids,
    dtw_impl,
    series1_preprocess=None,
    series1_sampling_rate=None,
    series2_preprocess=None,
    series2_sampling_rate=None,
):
    '''Compute DTW distances between two series'''
    
    dummy_preprocess = lambda data, _: data
    
    if series1_preprocess is None:
        series1_preprocess = dummy_preprocess
        
    if series2_preprocess is None:
        series2_preprocess = dummy_preprocess
    
    series1_length = len(series1_ids)
    series2_length = len(series2_ids)
    
    dtw_matrix = np.zeros((series1_length, series2_length))
    
    for i, series1_id in enumerate(series1_ids):
        for j, series2_id in enumerate(series2_ids):
            distance = dtw_distance(
                series1_preprocess(
                    series1[series1_id][series1_indices[series1_id][0]:series1_indices[series1_id][1]],
                    series1_sampling_rate
                ),
                series2_preprocess(
                    series2[series2_id][series2_indices[series2_id][0]:series2_indices[series2_id][1]],
                    series2_sampling_rate,
                ),
                dtw_impl,
            )

            dtw_matrix[i, j] = distance
                
    return dtw_matrix

def extract_training_features(config):
    '''Compute features from training feature set'''

    start_ts = datetime.datetime.now()
    
    user_id = config['user_id']
    dataset_path = config['database_path']
    dataset_type = config['database_type']
    
    stft_window_length = config['stft_window_length']
    stft_window_overlap = config['stft_window_overlap']
    stft_nfft = config['stft_nfft']
    
    activity_threshold = config['activity_threshold']
    activity_extra_samples = config['activity_extra_samples']
    activity_min_length = config['activity_min_length']

    dtw_impl = config['dtw_impl']
    
    trial_set = EmgTrialSet(dataset_path, user_id, dataset_type)
    trial_set_labels = trial_set.get_all_trials_labels()
    
    stft_window = scipy.signal.windows.hamming(stft_window_length)
    
    activity_indices, preprocessed_data, data = get_activity_indices_from_trial_set(
        trial_set,
        stft_window=stft_window,
        stft_overlap=stft_window_overlap,
        stft_nfft=stft_nfft,
        threshold=activity_threshold,
        extra_samples=activity_extra_samples,
        min_length=activity_min_length,
        get_activity_indices=get_activity_indices,
        preprocess=preprocess,
    )
    
    class_centers = {}
    
    for gesture in trial_set.labels:
        class_centers[gesture] = get_class_center_trial_id(
            preprocessed_data,
            activity_indices,
            trial_set_labels,
            gesture,
            dtw_impl,
        )
    
    class_centers_trial_ids = [class_centers[key] for key in class_centers]
    
    dtw_matrix = dtw_from_specific_trial_ids(
        preprocessed_data,
        activity_indices,
        class_centers_trial_ids,
        dtw_impl,
    )
    
    mean = dtw_matrix.mean(axis=1)
    std = dtw_matrix.std(axis=1, ddof=1)
    dtw_matrix_normalized = ((dtw_matrix.T - mean) / std).T
    
    end_ts = datetime.datetime.now()
        
    result = {
        'processing_time': end_ts - start_ts,
        'features': dtw_matrix_normalized,
        'labels': trial_set_labels,
        'preprocessed_data': preprocessed_data,
        'activity_indices': activity_indices,
        'class_centers': class_centers,
    }
    
    return result

def extract_test_features(config):
    '''Compute features from test feature set'''

    start_ts = datetime.datetime.now()
    
    user_id = config['user_id']
    dataset_path = config['database_path']
    dataset_type = config['database_type']
    
    stft_window_length = config['stft_window_length']
    stft_window_overlap = config['stft_window_overlap']
    stft_nfft = config['stft_nfft']
    
    activity_threshold = config['activity_threshold']
    activity_extra_samples = config['activity_extra_samples']
    activity_min_length = config['activity_min_length']

    dtw_impl = config['dtw_impl']
    
    feature_window_length = config['feature_window_length']
    feature_overlap_length = config['feature_overlap_length']
    
    training_data = config['training_data']
    training_preprocessed_data = training_data['preprocessed_data']
    training_activity_indices = training_data['activity_indices']
    training_class_centers = training_data['class_centers']

    trial_set = EmgTrialSet(dataset_path, user_id, dataset_type)
    trial_set_labels = trial_set.get_all_trials_labels()
    sampling_rate = trial_set.sampling_rate
    
    stft_window = scipy.signal.windows.hamming(stft_window_length)
    
    test_activity_indices, test_windows_preprocessed, test_windows = get_activity_indices_from_trial_set_windows(
        trial_set=trial_set,
        window_length=feature_window_length,
        overlap_length=feature_overlap_length,
        stft_window=stft_window,
        stft_overlap=stft_window_overlap,
        stft_nfft=stft_nfft,
        threshold=activity_threshold,
        extra_samples=activity_extra_samples,
        min_length=activity_min_length,
        get_activity_indices=get_activity_indices,
        preprocess=preprocess,
    )
    
    training_class_centers_trial_ids = [training_class_centers[key] for key in training_class_centers]
    
    trial_set_windows_dtw_matrix = np.zeros((
        test_activity_indices.shape[0],
        test_activity_indices.shape[1],
        len(training_class_centers_trial_ids),
    ))
    
    for test_trial_id, preprocessed_data in enumerate(test_windows_preprocessed):
        dtw_matrix = dtw_between_two_series(
            series1=test_windows[test_trial_id],
            series1_indices=test_activity_indices[test_trial_id],
            series1_ids=np.arange(0, test_windows.shape[1]),
            series2=training_preprocessed_data,
            series2_indices=training_activity_indices,
            series2_ids=training_class_centers_trial_ids,
            series1_preprocess=preprocess,
            series1_sampling_rate=sampling_rate,
            dtw_impl=dtw_impl,
        )
    
        mean = dtw_matrix.mean(axis=1)
        std = dtw_matrix.std(axis=1, ddof=1)
    
        dtw_matrix_normalized = ((dtw_matrix.T - mean) / std).T
    
        trial_set_windows_dtw_matrix[test_trial_id,:,:] = dtw_matrix_normalized
        
    end_ts = datetime.datetime.now()
        
    return {
        'processing_time': end_ts - start_ts,
        'features': trial_set_windows_dtw_matrix,
        'labels': trial_set_labels,
        'activity_indices': test_activity_indices,
    }
