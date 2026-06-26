import os
def save_pickle(obj, path):

    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path):

    with open(path, "rb") as f:
        obj = pickle.load(f)

    return obj
def create_directory(path):

    if not os.path.exists(path):
        os.makedirs(path)
def print_model_name(model):

    print(type(model).__name__)
def print_training_time(time_taken):

    print(f"Training Time : {time_taken:.4f} seconds")