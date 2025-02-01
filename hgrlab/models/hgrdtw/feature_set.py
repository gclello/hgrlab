import os
import json
import hashlib
import numpy as np
import h5py

from .feature_extraction import extract_training_features
from .feature_extraction import extract_test_features

class FeatureSetError(Exception):
    '''Error class for FeatureSet'''

    def __init__(self, message, details):
        super().__init__(message)
        
        self.message = message
        self.details = details

    def __reduce__(self):
        return (FeatureSetError, (self.message, self.details))

class FeatureSet:
    '''Collection of features from multiple users'''

    FILE_EXTENSION = '.h5'

    USER_PREFIX = 'subject'
    DS_TYPE_TRAINING = 'training'
    DS_TYPE_TEST = 'test'

    HDF5_DS_DTW_KEY = 'dtw'
    HDF5_DS_PREDICTED_INDICES_KEY = 'predicted_indices'
    HDF5_DS_LABELS_KEY = 'labels'
    HDF5_DS_INDICES_KEY = 'indices'
    HDF5_ATTR_CENTERS_KEY = 'centers'

    COMPRESSION_FILTER = 'gzip'
    COMPRESSION_LEVEL = 9

    @staticmethod
    def build_and_extract(config):
        fs = FeatureSet(**config)
        fs.extract_features()
        return fs

    def __init__(
            self,
            ds_name,
            ds_type,
            ds_dir,
            fs_dir,
            user_id,
            activity_threshold,
            activity_extra_samples=25,
            activity_min_length=100,
            stft_window_length=25,
            stft_window_overlap=10,
            stft_nfft=50,
            feature_window_length=500,
            feature_overlap_length=490,
        ):
            if not os.path.isdir(ds_dir):
                raise FeatureSetError(
                    message='Argument "%s" with value "%s" must be an existing directory' % ('ds_dir', ds_dir),
                    details={'code': 'Invalid directory', 'path': ds_dir, 'argument': 'ds_dir'}
                )
            
            self.ds_name = ds_name
            self.ds_type = ds_type
            self.ds_dir = ds_dir
            self.fs_dir = fs_dir
            self.user_id = user_id
            self.activity_threshold = activity_threshold
            self.activity_extra_samples = activity_extra_samples
            self.activity_min_length = activity_min_length
            self.stft_window_length = stft_window_length
            self.stft_window_overlap = stft_window_overlap
            self.stft_nfft = stft_nfft
            self.feature_window_length = feature_window_length
            self.feature_overlap_length = feature_overlap_length

    def __str__(self):
        config = {
            'ds_name': self.ds_name,
            'ds_type': self.ds_type,
            'ds_dir': self.ds_dir,
            'fs_dir': self.fs_dir,
            'user_id': self.user_id,
            'activity_threshold': self.activity_threshold,
            'activity_extra_samples': self.activity_extra_samples,
            'activity_min_length': self.activity_min_length,
            'stft_window_length': self.stft_window_length,
            'stft_window_overlap': self.stft_window_overlap,
            'stft_nfft': self.stft_nfft,
            'feature_window_length': self.feature_window_length,
            'feature_overlap_length': self.feature_overlap_length,
        }

        return '[EMG Feature Set] %s' % config
    
    def get_hash(self):
        h = hashlib.md5()
        h.update(self.ds_name.encode(encoding='utf-8'))
        h.update(self.ds_type.encode(encoding='utf-8'))
        h.update(str(self.user_id).encode(encoding='utf-8'))
        h.update(str(self.activity_threshold).encode(encoding='utf-8'))
        h.update(str(self.activity_extra_samples).encode(encoding='utf-8'))
        h.update(str(self.activity_min_length).encode(encoding='utf-8'))
        h.update(str(self.stft_window_length).encode(encoding='utf-8'))
        h.update(str(self.stft_window_overlap).encode(encoding='utf-8'))
        h.update(str(self.stft_nfft).encode(encoding='utf-8'))
        h.update(str(self.feature_window_length).encode(encoding='utf-8'))
        h.update(str(self.feature_overlap_length).encode(encoding='utf-8'))
        return h.hexdigest()
    
    def get_features_file_name(self):
        return '{DS_NAME}_features_{USER_PREFIX}{USER_ID}_{DS_TYPE}_{HASH}{FILE_EXTENSION}'.format(
            DS_NAME=self.ds_name,
            USER_PREFIX=self.USER_PREFIX,
            USER_ID=self.user_id,
            DS_TYPE=self.ds_type,
            HASH=self.get_hash(),
            FILE_EXTENSION=FeatureSet.FILE_EXTENSION,
        )
    
    def get_features_file_path(self):
        return os.path.join(self.fs_dir, self.get_features_file_name())

    def extract_features(self):
        path = self.get_features_file_path()
        if os.path.isfile(path):
            return True
        
        config = {
            'user_id': self.user_id,
            'database_path': self.ds_dir,
            'database_type': 'training',
            'activity_threshold': self.activity_threshold,
            'activity_extra_samples': self.activity_extra_samples,
            'activity_min_length': self.activity_min_length,
            'stft_window_length': self.stft_window_length,
            'stft_window_overlap': self.stft_window_overlap,
            'stft_nfft': self.stft_nfft,
        }

        training_data = extract_training_features(config)
        
        if self.ds_type == self.DS_TYPE_TEST:
            config['database_type'] = 'test'
            config['feature_window_length'] = self.feature_window_length
            config['feature_overlap_length'] = self.feature_overlap_length
            config['training_data'] = training_data

            test_data = extract_test_features(config)
        else:
            test_data  = None

        self.save_data(training_data, test_data)

        return False
    
    def create_features_file(self):
        path = self.get_features_file_path()

        if not os.path.isdir(self.fs_dir):
            os.mkdir(self.fs_dir)

        with h5py.File(path, 'w-') as f:
            f.attrs['hash'] = self.get_hash()
            f.attrs['ds_name'] = self.ds_name
            f.attrs['ds_type'] = self.ds_type
            f.attrs['user_id'] = self.user_id
            f.attrs['activity_threshold'] = self.activity_threshold
            f.attrs['activity_extra_samples'] = self.activity_extra_samples
            f.attrs['activity_min_length'] = self.activity_min_length
            f.attrs['stft_window_length'] = self.stft_window_length
            f.attrs['stft_window_overlap'] = self.stft_window_overlap
            f.attrs['stft_nfft'] = self.stft_nfft
            f.attrs['feature_window_length'] = self.feature_window_length
            f.attrs['feature_overlap_length'] = self.feature_overlap_length

        return path
    
    def save_data(self, training_data, test_data=None):
        path = self.create_features_file()

        labels_dtype = h5py.special_dtype(vlen=str)
        longest_label_size = 0

        training_centers = training_data['class_centers']

        data = test_data if self.ds_type == self.DS_TYPE_TEST else training_data

        features = data['features']
        predicted_indices = data['activity_indices']

        if 'labels' in data.keys():
            labels = data['labels']
            longest_label_size = len(max(labels, key=len))
            if longest_label_size > longest_label_size:
                longest_label_size = longest_label_size
        else:
            labels = None

        if 'indices' in data.keys():
            indices = data['indices']
        else:
            indices = None

        class FeaturesJsonEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                return super(FeaturesJsonEncoder, self).default(obj)

        with h5py.File(path, 'r+') as f:
            f.attrs[FeatureSet.HDF5_ATTR_CENTERS_KEY] = \
                json.dumps(training_centers, cls=FeaturesJsonEncoder)

            f.create_dataset(
                FeatureSet.HDF5_DS_DTW_KEY,
                data=features,
                chunks=features.shape,
                compression=FeatureSet.COMPRESSION_FILTER,
                compression_opts=FeatureSet.COMPRESSION_LEVEL,
            )

            f.create_dataset(
                FeatureSet.HDF5_DS_PREDICTED_INDICES_KEY,
                data=predicted_indices,
                chunks=predicted_indices.shape,
                compression=FeatureSet.COMPRESSION_FILTER,
                compression_opts=FeatureSet.COMPRESSION_LEVEL,
            )

            if labels is not None:
                f.create_dataset(
                    FeatureSet.HDF5_DS_LABELS_KEY,
                    data=np.array(labels, dtype='S'+str(longest_label_size)),
                    dtype=labels_dtype,
                    chunks=labels.shape,
                    compression=FeatureSet.COMPRESSION_FILTER,
                    compression_opts=FeatureSet.COMPRESSION_LEVEL,
                )

            if indices is not None:
                f.create_dataset(
                    FeatureSet.HDF5_DS_INDICES_KEY,
                    data=indices,
                    chunks=indices.shape,
                    compression=FeatureSet.COMPRESSION_FILTER,
                    compression_opts=FeatureSet.COMPRESSION_LEVEL,
                )

    def get_data(self, key):
        with h5py.File(self.get_features_file_path(), 'r') as f:
            if key == FeatureSet.HDF5_DS_LABELS_KEY:
                return np.array(f[key]).astype(str)
            return np.array(f[key])
