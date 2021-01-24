import pickle
def save_weights(model, path):
    pickle.dump(model, open(path, 'wb'))

def load_weights(path):
    return pickle.load(open(path, 'rb'))
