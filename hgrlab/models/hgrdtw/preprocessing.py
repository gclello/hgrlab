import numpy as np
import scipy.signal

from hgrlab.utils import sliding_window

def preprocess(data, sampling_rate, filter_order=5, cutoff_frequency=10):
    '''Rectify and filter input data'''
    
    sos = scipy.signal.butter(
        filter_order, 
        cutoff_frequency,
        'lowpass',
        fs=sampling_rate,
        output='sos'
    )

    if data.ndim > 1 and data.shape[-2] > data.shape[-1]:
        return np.swapaxes(scipy.signal.sosfiltfilt(sos, np.abs(np.swapaxes(data,-2,-1))),-2,-1)
    else:
        return scipy.signal.sosfiltfilt(sos, np.abs(data))

def preprocess_trial_set(trial_set):
    '''Preprocess each trial in trial set'''

    trial_set_data = trial_set.get_all_trials()
    preprocessed_data = np.zeros_like(trial_set_data)

    for trial_id, trial_data in enumerate(trial_set_data):
        data_length = trial_set.get_trial_data_length(trial_id)
        preprocessed_data[trial_id,0:data_length,:] = preprocess(trial_data[0:data_length,:])
        
    return preprocessed_data

def preprocess_trial_set_windows(trial_set, window_length, overlap_length):
    '''Preprocess each trial window in trial set'''

    trial_set_data = trial_set.get_all_trials()
    
    trial_set_windows = sliding_window(
        trial_set_data,
        window_length=window_length,
        overlap_length=overlap_length,
        axis=1,
    )

    return preprocess(trial_set_windows)
