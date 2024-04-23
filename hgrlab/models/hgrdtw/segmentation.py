import numpy as np
import scipy.signal

from hgrlab.utils import sliding_window

def segment(
    data,
    sampling_rate,
    threshold,
    stft_window_length=25,
    stft_overlap=10,
    stft_nfft=50,
    extra_samples=25,
    min_length=100
):
    '''Segment data by returning the start and end indices of the event of interest'''
    
    stft_window = scipy.signal.windows.hamming(stft_window_length)
    
    return get_activity_indices(
        data,
        sampling_rate,
        stft_window,
        stft_overlap,
        stft_nfft,
        threshold,
        extra_samples,
        min_length
    )

def get_activity_indices(
    data,
    sampling_rate,
    stft_window,
    stft_overlap,
    stft_nfft,
    threshold,
    extra_samples,
    min_length
):
    '''Get start and end indices between which activity is higher than the specified threshold'''
    
    data_length = data.shape[0]
    channels_sum = data.sum(axis=1)
    stft_window_size = len(stft_window)
    
    f,t,stft = scipy.signal.stft(
        channels_sum,
        fs=sampling_rate,
        window=stft_window,
        nperseg=stft_window_size,
        noverlap=stft_overlap,
        nfft=stft_nfft,
        boundary=None,
        padded=False
    )
    
    stft_abs = ((stft_window_size + 1) / 2) * np.abs(stft)
    stft_sum = stft_abs.sum(axis=0)
    
    if len(stft_sum) <= 2:
        raise Exception('The number of stft windows is too short!')
    
    stft_indices_above_threshold = stft_sum >= threshold
    stft_indices_with_transitions = np.diff(stft_indices_above_threshold, prepend=False, append=False)
    
    if stft_indices_with_transitions[-1]:
        stft_indices_with_transitions[-2] = stft_indices_with_transitions[-1]
        
    stft_indices_with_transitions = stft_indices_with_transitions[0:-1]
    
    transitions_indices = np.nonzero(stft_indices_with_transitions == True)[0]
    number_of_transitions = len(transitions_indices)
    
    stft_time_indices = np.floor(t * sampling_rate).astype(int) - 1
    
    start_index = 0
    end_index = data_length
    
    if number_of_transitions == 1:
        stft_start_index = transitions_indices[0]
        start_index = stft_time_indices[stft_start_index]
    elif number_of_transitions >= 2:
        stft_start_index = transitions_indices[0]
        stft_end_index = transitions_indices[-1] - 1
        
        start_index = stft_time_indices[stft_start_index]
        end_index = stft_time_indices[stft_end_index] + 1
    
    start_index = np.maximum(0, start_index - extra_samples)
    end_index = np.minimum(data_length, end_index + extra_samples)
    
    if (end_index - start_index) < min_length:
        start_index = 0
        end_index = data_length
        
    return (start_index, end_index)

def get_activity_indices_from_trial_set(
        trial_set,
        stft_window,
        stft_overlap,
        stft_nfft,
        threshold,
        extra_samples,
        min_length,
        get_activity_indices,
        preprocess=None,
):
    '''Get activity indices for each trial in trial set'''
    
    trial_set_data = trial_set.get_all_trials()
    sampling_rate = trial_set.sampling_rate 
    
    preprocessed_data = np.zeros_like(trial_set_data)
    activity_indices = np.zeros((trial_set_data.shape[0], 2), 'int')
    
    if preprocess is None:
        preprocessed_data = trial_set_data
        
    for trial_id, trial_data in enumerate(trial_set_data):
        data_length = trial_set.get_trial_data_length(trial_id)
        
        if preprocess is not None:
            preprocessed_data[trial_id,0:data_length,:] = preprocess(
                trial_data[0:data_length,:],
                sampling_rate
            )
        
        start_index, end_index = get_activity_indices(
            preprocessed_data[trial_id,0:data_length,:],
            sampling_rate,
            stft_window,
            stft_overlap,
            stft_nfft,
            threshold,
            extra_samples,
            min_length
        )
        
        activity_indices[trial_id][0] = start_index
        activity_indices[trial_id][1] = end_index
        
    return activity_indices, preprocessed_data, trial_set_data

def get_activity_indices_from_trial_set_windows(
        trial_set,
        window_length,
        overlap_length,
        stft_window,
        stft_overlap,
        stft_nfft,
        threshold,
        extra_samples,
        min_length,
        get_activity_indices,
        preprocess=None,
):
    '''Get activity indices for each trial window in trial set'''
    
    trial_set_data = trial_set.get_all_trials()
    sampling_rate = trial_set.sampling_rate
    
    trial_set_windows = sliding_window(
        trial_set_data,
        window_length=window_length,
        overlap_length=overlap_length,
        axis=1,
    )
    
    trial_set_windows_activity_indices = np.zeros((
        trial_set_windows.shape[0],
        trial_set_windows.shape[1],
        2
    ), 'int')
    
    if preprocess is None:
        preprocessed_data = trial_set_windows
    else:
        preprocessed_data = preprocess(
            trial_set_windows,
            sampling_rate
        )
        
    for trial_id, trial_windows in enumerate(trial_set_windows):
        data_length = trial_set.get_trial_data_length(trial_id)
        
        for window_id, _ in enumerate(trial_windows):
            start_index, end_index = get_activity_indices(
                preprocessed_data[trial_id,window_id,:data_length,:],
                sampling_rate,
                stft_window,
                stft_overlap,
                stft_nfft,
                threshold,
                extra_samples,
                min_length
            )
            
            trial_set_windows_activity_indices[trial_id][window_id][0] = start_index
            trial_set_windows_activity_indices[trial_id][window_id][1] = end_index
            
    return trial_set_windows_activity_indices, preprocessed_data, trial_set_windows
