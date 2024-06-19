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
        encoded_data = []
        
        samples = X.shape[0]
        features = X.shape[1]
        
        for i in np.arange(0, samples):
            encoded_data.append([])
            
            for j in np.arange(0, features):
                encoded_value = self.thermometers[j].encode(X[i][j])
                encoded_data[i].extend(encoded_value)
        
        return np.array(encoded_data)
