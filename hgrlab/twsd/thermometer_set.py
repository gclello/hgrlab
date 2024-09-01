import numpy as np
from .thermometer import Thermometer

class ThermometerSet:
    def __init__(self):
        self.thermometers = []
        
    def calibrate(self, X, *args, **kwargs):
        self.thermometers = []
        
        features = X.shape[1]
        
        for i in np.arange(0, features):
            t = Thermometer(*args, **kwargs)
            t.calibrate(X[:,i])
            self.thermometers.append(t)
    
    def encode(self, X):
        samples = X.shape[0]
        features = X.shape[1]

        encoded_data = np.empty((samples, 0), dtype=np.uint8)
        
        for i in np.arange(0, features):
            encoded_feature = self.thermometers[i].encode(X[:,i])
            encoded_data = np.concatenate([encoded_data, encoded_feature], axis=1)
        
        return encoded_data
