import pickle

def pickle_data(data, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)
    print(f'Data successfully pickled to {file_path}')

# Function to unpickle data
def unpickle_data(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    print(f'Data successfully unpickled from {file_path}')
    return data