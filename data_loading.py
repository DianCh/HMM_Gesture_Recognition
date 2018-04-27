import numpy as np
from sklearn.cluster import KMeans
import os


def discretize_raw(folder, M, classifier=None):
    # Construct a dictionary for storing different sequences and observations
    sequences = {}
    obs_sequences = {}

    # Load raw data and get rid of timestamps
    X = np.zeros((0, 6))

    for root, dirs, files in os.walk(folder):
        for file in files:
            # Avoid any hidden files that's not our intended data
            if ".txt" not in file:
                continue

            name = file.split(".txt")[0]
            raw_sequence = np.loadtxt(os.path.join(root, file))[:, 1:]

            # Apply low pass filter to the raw data
            filtered = low_pass(raw_sequence)

            sequences[name] = filtered

            X = np.concatenate((X, raw_sequence), axis=0)

    if classifier is None:
        # Train a K-Means classifier using all training data
        classifier = KMeans(n_clusters=M, random_state=0).fit(X)

    for name, sequence in sequences.items():
        obs_sequences[name] = classifier.predict(sequence)

    return classifier, obs_sequences


def low_pass(raw_sequence):
    # Coefficient of low pass filter
    alpha = 0.7

    filtered = np.zeros(raw_sequence.shape)
    filtered[0, :] = raw_sequence[0, :]

    for i in range(1, raw_sequence.shape[0]):
        filtered[i, :] = alpha * raw_sequence[i, :] + (1 - alpha) * filtered[i - 1, :]

    return filtered

