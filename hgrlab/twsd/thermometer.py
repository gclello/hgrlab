import numpy as np

class Thermometer:
    _strategies = [
        "linear"
    ]
    
    _default_strategy = "linear"
    _default_size = 8
    
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

    @staticmethod
    def encode_uint_vector(x):
        encoded_vector = np.empty((len(x), 0), dtype=np.uint8)
    
        while x.any():
            encoded_byte = np.fliplr(np.unpackbits((1 << x) - 1).reshape(-1,8))
            encoded_vector = np.concatenate([encoded_vector, encoded_byte], axis=1)
            x = np.where(x<8, 0, x-8)
        
        return encoded_vector
    
    @staticmethod
    def encode_uint_vector_fixed_size(x, size):
        encoded_vector = np.empty((len(x), 0), dtype=np.uint8)
        number_of_bytes = np.ceil(size // 8)
    
        for i in np.arange(0, number_of_bytes):
            encoded_byte = np.fliplr(np.unpackbits((1 << x) - 1).reshape(-1,8))
            encoded_vector = np.concatenate([encoded_vector, encoded_byte], axis=1)
            if x.any():
                x = np.where(x<8, 0, x-8)
        
        return encoded_vector
    
    def encode(self, X):
        X_steps = (X - self.min) // self.step
        X_steps[X_steps > self.size] = self.size
        X_steps[X_steps < 0] = 0

        return Thermometer.encode_uint_vector_fixed_size(
            X_steps.astype('uint8', copy=False),
            size=self.size,
        )
