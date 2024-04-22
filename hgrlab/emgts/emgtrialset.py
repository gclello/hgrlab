import os
import h5py

class EmgTrialSetError(Exception):
    '''Error class for EmgTrialSet'''

    def __init__(self, message, details):
        super().__init__(message)
        
        self.message = message
        self.details = details

    def __reduce__(self):
        return (EmgTrialSetError, (self.message, self.details))
    
def get_filename(user_id, dataset_type):
    return 'subject%d_%s%s' % (
        user_id,
        dataset_type,
        EmgTrialSet.FILE_EXTENSION,
    )

def get_path(base_dir, user_id, dataset_type):
    return os.path.join(
        base_dir,
        get_filename(user_id, dataset_type),
    )

class EmgTrialSet:
    '''Collection of sEMG recording trials from a single user'''

    FILE_EXTENSION = '.h5'
    EXTENSION_INDEX = len(FILE_EXTENSION) * -1

    HDF5_DS_DATA_KEY = 'data'
    HDF5_DS_DATA_LENGTH_KEY = 'data_length'
    HDF5_DS_LABELS_KEY = 'labels'
    
    HDF5_ATTR_SAMPLING_RATE_KEY = 'sampling_rate'
    HDF5_ATTR_LABELS_KEY = 'labels'
    
    def __init__(self):
        self.path = ''
        self.user_id = 0
        self.dataset_type = ''
        self.sampling_rate = 0
        self.labels = []

    def __init__(self, base_dir, user_id, dataset_type):
        self.load_trial_set(base_dir, user_id, dataset_type)
        
    def __str__(self):
        return '[EMG Trial Set] User ID = %d, DS Type = %s, sampling_rate = %d Hz, labels = %s, filename = %s' % (
            self.user_id,
            self.dataset_type,
            self.sampling_rate,
            self.labels,
            os.path.basename(self.path)
        )
    
    def load_trial_set(self, base_dir, user_id, dataset_type):
        path = get_path(base_dir, user_id, dataset_type)
        
        with h5py.File(path, 'r') as f:
            if (EmgTrialSet.HDF5_DS_DATA_KEY in f.keys()) is False:
                raise EmgTrialSetError(
                    message='EmgTrialSet is missing dataset "%s" (%s)' % (EmgTrialSet.HDF5_DS_DATA_KEY, path),
                    details={'code': 'Invalid DS key', 'path': path, 'key': EmgTrialSet.HDF5_DS_DATA_KEY}
                )
                
            if (EmgTrialSet.HDF5_DS_DATA_LENGTH_KEY in f.keys()) is False:
                raise EmgTrialSetError(
                    message='EmgTrialSet is missing dataset "%s" (%s)' % (EmgTrialSet.HDF5_DS_DATA_LENGTH_KEY, path),
                    details={'code': 'Invalid DS key', 'path': path, 'key': EmgTrialSet.HDF5_DS_DATA_LENGTH_KEY}
                )
                
            if (EmgTrialSet.HDF5_DS_LABELS_KEY in f.keys()) is False:
                raise EmgTrialSetError(
                    message='EmgTrialSet is missing dataset "%s" (%s)' % (EmgTrialSet.HDF5_DS_LABELS_KEY, path),
                    details={'code': 'Invalid DS key', 'path': path, 'key': EmgTrialSet.HDF5_DS_LABELS_KEY}
                )
                
            if (EmgTrialSet.HDF5_ATTR_SAMPLING_RATE_KEY in f.attrs.keys()) is False:
                raise EmgTrialSetError(
                    message='EmgTrialSet is missing attribute "%s" (%s)' % (EmgTrialSet.HDF5_ATTR_SAMPLING_RATE_KEY, path),
                    details={'code': 'Invalid attribute key', 'path': path, 'key': EmgTrialSet.HDF5_ATTR_SAMPLING_RATE_KEY}
                )
                
            if (EmgTrialSet.HDF5_ATTR_LABELS_KEY in f.attrs.keys()) is False:
                raise EmgTrialSetError(
                    message='EmgTrialSet is missing attribute "%s" (%s)' % (EmgTrialSet.HDF5_ATTR_LABELS_KEY, path),
                    details={'code': 'Invalid attribute key', 'path': path, 'key': EmgTrialSet.HDF5_ATTR_LABELS_KEY}
                )
            
            self.path = path
            self.user_id = user_id
            self.dataset_type = dataset_type
            self.sampling_rate = f.attrs[EmgTrialSet.HDF5_ATTR_SAMPLING_RATE_KEY]
            self.labels = f.attrs[EmgTrialSet.HDF5_ATTR_LABELS_KEY]
            
    def get_trial(self, trial_id):
        with h5py.File(self.path, 'r') as f:            
            return f[EmgTrialSet.HDF5_DS_DATA_KEY][trial_id, :, :]
        
    def get_trial_data_length(self, trial_id):
        with h5py.File(self.path, 'r') as f:            
            return f[EmgTrialSet.HDF5_DS_DATA_LENGTH_KEY][trial_id]
        
    def get_trial_label(self, trial_id):
        with h5py.File(self.path, 'r') as f:            
            return f[EmgTrialSet.HDF5_DS_LABELS_KEY][trial_id].decode("utf-8")
        
    def get_all_trials(self):
        with h5py.File(self.path, 'r') as f:            
            return f[EmgTrialSet.HDF5_DS_DATA_KEY][:, :, :]
        
    def get_all_trials_data_length(self):
        with h5py.File(self.path, 'r') as f:            
            return f[EmgTrialSet.HDF5_DS_DATA_LENGTH_KEY][:]
        
    def get_all_trials_labels(self):
        with h5py.File(self.path, 'r') as f:            
            return f[EmgTrialSet.HDF5_DS_LABELS_KEY][:].astype(str)
