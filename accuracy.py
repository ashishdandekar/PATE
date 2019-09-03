import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from utilities import *

if __name__ == "__main__":
    t = time.time()
    test_x, test_y = loadmnist('data/t10k-images-idx3-ubyte',
            'data/t10k-labels-idx1-ubyte')
    print("Time to load datasets: ", time.time() - t)

    accuracy_teachers = []
    for n_teachers in [50, 100, 150, 250]:
        models = pickle.load(open("teachers_%d.pickle" % n_teachers, "rb"))
        accuracy_teachers.append([model.score(test_x, test_y) for model in models])
        print("Done: ", n_teachers)

    plt.boxplot(accuracy_teachers)
    plt.xticks([1, 2, 3, 4], ["50", "100", "150", "250"])
    plt.xlabel("Number of teachers")
    plt.ylabel("Accuracy")
    plt.savefig("teacher_accuracy.png", bbox_inches = "tight")
