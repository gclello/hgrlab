import numpy as np

class Thermometer:
    _strategies = [
        "linear"
    ]
    
    _default_strategy = "linear"
    _default_size = 2 ** 8
    
    def __init__(self, *args, **kwargs):
        self.size = int(kwargs.get("size", Thermometer._default_size))
        strategy = kwargs.get("strategy", Thermometer._default_strategy)
        
        if(strategy in Thermometer._strategies):
            self.strategy = strategy
        else:
            self.strategy = Thermometer._default_strategy
        
        self.min = 0
        self.max = 0
        self.step = 0
    
    def calibrate(self, X):
        if(self.strategy == "linear"):
            self.calibrate_linear(X)
        
    def calibrate_linear(self, X):
        self.min = X.min()
        self.max = X.max()
        self.step = (self.max - self.min) / self.size
    
    def encode(self, sample):
        number_of_steps = int((sample - self.min) // self.step)
        if(number_of_steps > self.size):
            number_of_steps = self.size
        if(number_of_steps < 0):
            number_of_steps = 0
        
        encoded_sample = np.zeros((self.size), dtype=int)
        encoded_sample[0:number_of_steps] = np.ones(number_of_steps)
        
        return encoded_sample
