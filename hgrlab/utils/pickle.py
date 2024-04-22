import pickle

def save_pickle(path, data):
    '''Save data to pickle file'''
    
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        
def load_pickle(path):
    '''Load data from pickle file'''

    data = None

    with open(path, 'rb') as f:
        data = pickle.load(f)

    return data
