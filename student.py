import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from utilities import *
from collections import defaultdict
from sklearn.linear_model import LogisticRegression

def build_model(x, y):
    return LogisticRegression(solver = 'lbfgs').fit(x, y)

def generate_dataset(x, models, noise_eps):
    # This parts implements noisy max mechanism
    N_classes = len(models[0].classes_)
    predictions = np.array([model.predict(x) for model in models]).T

    y = []
    for datapt in predictions:
        votes = np.zeros(N_classes)
        for vote in datapt: votes[vote] += 1

        votes += np.random.laplace(0, 1. / noise_eps, N_classes)

        y.append(np.argmax(votes))

    return (x, y)

if __name__ == "__main__":
    t = time.time()
    test_x, test_y = loadmnist('data/t10k-images-idx3-ubyte',
            'data/t10k-labels-idx1-ubyte')
    print("Time to load datasets: ", time.time() - t)

    validation_x, validation_y = test_x[5000:], test_y[5000:]

    teachers = {}
    n_teachers = [50, 100, 150, 250]
    for n in n_teachers:
        teachers[n] = pickle.load(open("teachers_%d.pickle" % n, "rb"))

    n_queries = np.arange(0, 500, 50) + 200

    N_EXPTS = 50
    noise_eps = 0.1

    # Query analysis
    # We fix number of teachers to 250
    # query_analysis = defaultdict(list)
    # for i in range(N_EXPTS):
    #     for q in n_queries:
    #         X = test_x[np.random.choice(5000, q, replace = False), :]
    #         synth_dataset = generate_dataset(X, teachers[250], noise_eps)
    #         query_analysis[q].append(
    #                 build_model(*synth_dataset).score(validation_x, validation_y))
    #     print("Finished iteration: ", i)
    # plt.clf()
    # plt.boxplot([query_analysis[q] for q in n_queries])
    # plt.xticks(np.array(range(len(n_queries))) + 1, map(str, n_queries))
    # plt.xlabel("Number of queries")
    # plt.ylabel("Accuracy")
    # plt.title("With 250 teachers")
    # plt.savefig("student_query_accuracy.png", bbox_inches = "tight")

    # teacher analysis
    # We fix number of 
    teacher_analysis = defaultdict(list)
    for i in range(N_EXPTS):
        X = test_x[np.random.choice(5000, 500, replace = False), :]
        for t in n_teachers:
            synth_dataset = generate_dataset(X, teachers[t], noise_eps)
            teacher_analysis[t].append(
                    build_model(*synth_dataset).score(validation_x, validation_y))
        print("Finished iteration: ", i)
    plt.clf()
    plt.boxplot([teacher_analysis[q] for q in n_teachers])
    plt.xticks(np.array(range(len(teachers))) + 1, map(str, n_teachers))
    plt.xlabel("Number of teachers")
    plt.ylabel("Accuracy")
    plt.title("With 500 queries")
    plt.savefig("student_teacher_accuracy.png", bbox_inches = "tight")
