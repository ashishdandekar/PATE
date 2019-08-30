import time
import pickle
import numpy as np
import multiprocessing as mp
from sklearn.linear_model import LogisticRegression
from utilities import *

N_TEACHERS = 100

def split_data(x, y):
    return zip(np.split(x, N_TEACHERS), np.split(y, N_TEACHERS))

def build_model(x, y):
    return LogisticRegression(solver = 'lbfgs').fit(x, y)

if __name__ == "__main__":
    t = time.time()
    train_x, train_y = loadmnist('data/train-images-idx3-ubyte',
            'data/train-labels-idx1-ubyte')
    test_x, test_y = loadmnist('data/t10k-images-idx3-ubyte',
            'data/t10k-labels-idx1-ubyte')
    print("Time to load datasets: ", time.time() - t)

    datasets = split_data(train_x, train_y)

    pool = mp.Pool(processes=4)
    t = time.time()
    models = [pool.apply(build_model, args=(x, y)) for x, y in datasets]
    print("Time for training: ", time.time() - t)

    accuracy = [model.score(test_x, test_y) for model in models]
    print(accuracy)

    votes = np.array([model.predict(test_x) for model in models]).T
    print("Votes shape: ", votes.shape)

    with open("teachers.pickle", "wb") as teachers, open("votes.pickle", "wb") as fvotes:
        pickle.dump(models, teachers)
        pickle.dump(votes, fvotes)
