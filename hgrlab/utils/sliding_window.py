import numpy as np

def sliding_window(x, window_length, overlap_length, axis=0):
    '''Create sliding window view from NumPy array'''

    n = x.shape[axis]
    m = window_length
    l = overlap_length
    k = (n - l) // (m - l)
    
    shape = x.shape[:axis] + (k, m) + x.shape[axis+1:]
    strides = x.strides[:axis] + (x.strides[axis] * (m - l), x.strides[axis]) + x.strides[axis+1:]
    
    return np.lib.stride_tricks.as_strided(
        x,
        shape=shape,
        strides=strides,
        writeable=False
    )
