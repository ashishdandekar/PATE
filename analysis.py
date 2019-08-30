import math
import pickle
import numpy as np
import matplotlib.pyplot as plt

def compute_q_noisy_max(counts, noise_eps):
    winner = np.argmax(counts)
    counts_normalized = noise_eps * (counts - counts[winner])
    counts_rest = np.array(
            [counts_normalized[i] for i in range(len(counts)) if i != winner])
    q = 0.0
    for c in counts_rest:
        gap = -c
        q += (2.0 + gap) / (4.0 * math.exp(gap))
    return min(q, 1.0 - (1.0 / len(counts)))

def logmgf_exact(q, priv_eps, l):
    if q < 0.5:
        t_one = (1 - q) * math.pow((1 - q) / (1 - (math.exp(priv_eps) * q)), l)
        t_two = q * math.exp(priv_eps * l)
        t = t_one + t_two
        try:
            log_t = math.log(t)
        except ValueError:
            log_t = priv_eps * l
    else:
        log_t = priv_eps * l

    # first one is the data_independent bound
    return min(0.5 * priv_eps * priv_eps * l * (l + 1), log_t, priv_eps * l)

def logmgf_from_counts(counts, noise_eps, l):
    q = compute_q_noisy_max(counts, noise_eps)
    return logmgf_exact(q, 2.0 * noise_eps, l)

if __name__ == "__main__":
    original_votes = pickle.load(open("votes.pickle", "rb"))

    noise_eps = 0.1 # epsilon for each call in noisy_max
    noise_delta = 1e-5
    l_moments = 1 + np.array(range(8))

    qrs = []
    eps = []
    for queries in np.arange(0, len(original_votes), 200) + 200:
        votes = original_votes[:queries]

        N, NUM_TEACHERS = votes.shape
        counts = np.zeros((N, 10))
        for i in range(N):
            for j in range(NUM_TEACHERS):
                counts[i, votes[i, j]] += 1

        total_log_mgf_nm = np.array([0.0 for _ in l_moments])

        for n in range(N):
            total_log_mgf_nm += np.array(
                    [logmgf_from_counts(counts[i], noise_eps, l) for l in l_moments])

        eps_list_nm = (total_log_mgf_nm - math.log(noise_delta)) / l_moments

        print("Queries:  ", queries)
        qrs.append(queries)
        eps.append(min(eps_list_nm))

    plt.plot(qrs, eps)
    plt.xlabel("Number of queries")
    plt.ylabel("$\epsilon$")
    plt.title("With 100 teachers")
    plt.savefig("queries_analysis.png", bbox_inches = "tight")
