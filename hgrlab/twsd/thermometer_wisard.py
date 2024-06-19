import numpy as np
import wisardpkg
from .thermometer_set import ThermometerSet

class ThermometerWisard:
    def __init__(self, address_size, *args, **kwargs):
        self.thermometer_set = None

        self.thermometer_size = kwargs.get("thermometer_size", 2 ** 8)
        model_as_json = kwargs.get("json", None)
        
        if(model_as_json is not None):
            self.address_size = None
            self.tuple_indexes = []
            self.model = wisardpkg.Wisard(model_as_json)
        else:
            self.address_size = address_size
            self.tuple_indexes = kwargs.get("tuple_indexes", [])
        
            self.model = wisardpkg.Wisard(
                self.address_size,
                bleachingActivated=True,
                ignoreZero=False,
                completeAddressing=True,
                verbose=False,
                indexes=self.tuple_indexes,
                base=2,
                confidence=1
            )
    
    @classmethod
    def from_json(cls, model_as_json):
        return cls(None, json=model_as_json)
                   
    def to_json(self):
        return self.model.json()
    
    def get_mental_images(self):
        return self.model.getMentalImages()
    
    def calibrate(self, X):
        if(self.thermometer_set is None):
            self.thermometer_set = ThermometerSet()
        
        self.thermometer_set.calibrate(
            X,
            size=self.thermometer_size,
        )
        
    def quantize(self, X):
        if(self.thermometer_set is None):
            self.calibrate(X)
        
        return self.thermometer_set.encode(X)
    
    def fit(self, X, y, quantize=True):
        string_labels = y.astype(str)
        
        if(quantize):
            input_data = self.quantize(X)
        else:
            input_data = X
        
        self.model.train(input_data, string_labels)
    
    def predict(self, X, quantize=True):
        if(quantize):
            input_data = self.quantize(X)
        else:
            input_data = X
        
        prediction = np.array(self.model.classify(input_data))
        return prediction
