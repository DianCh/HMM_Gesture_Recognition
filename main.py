import numpy as np
import pickle

from GestureModel import GestureModel
import GestureModel as GM
import data_loading as dl


def main(retrain=False, recluster=False, N=20, M=40):
    # Specify folders and file names
    train_folder = "train_data/"
    cross_folder = "cross_validation/"
    test_folder = "test_data/"

    model_file = "./models/models_" + str(N) + "_" + str(M) + ".pkl"
    kmeans_file = "./models/kmeans_" + str(M) + ".pkl"

    # Decide whether or not to recluster the raw data or load pre-trained classifier
    if recluster:
        kmeans, _ = dl.discretize_raw(train_folder, M, classifier=None)
        with open(kmeans_file, 'wb') as file:
            pickle.dump(kmeans, file)
    else:
        # Load the pre-trained K-Means classifier
        with open(kmeans_file, "rb") as file:
            kmeans = pickle.load(file)

    # Decide whether to retrain models or load pre-trained models
    if retrain:
        _, models = GM.train_models(train_folder, N, M, classifier=kmeans)
        with open(model_file, 'wb') as file:
            pickle.dump(models, file)
    else:
        # Load the pre-trained models
        with open(model_file, 'rb') as file:
            models = pickle.load(file)

    # Read in test data and start predicting
    _, obs_sequences = dl.discretize_raw(test_folder, M, kmeans)
    gestures = ["beat3", "beat4", "circle", "eight", "inf", "wave"]
    for name, obs_sequence in obs_sequences.items():
        print("sequence:", name)
        label, log_probs, probs, confidence_naive, confidence_timed = GM.predict(models, obs_sequence)
        print("Predicted gesture:", label)
        print("Confidence Naive: %.3f  Confidence Timed: %.3f" % (confidence_naive, confidence_timed))
        for i in range(6):
            print("Log prob for", gestures[i], ":", log_probs[i],
                  "Normalized Prob :", probs[i])
        print("---------------------------")
        GM.plot_probs(name, label, confidence_naive, confidence_timed, log_probs, probs)


    GM.plot_correlation(models, show_figure=False)

    return 0


if __name__ == "__main__":
    main(retrain=False, recluster=False, N=20, M=40)
