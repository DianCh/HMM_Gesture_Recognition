import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm as cm

import data_loading as dl


class GestureModel(object):

    def __init__(self, name, num_states, num_obs, empty=False):
        self.name = name

        self.N = num_states
        self.M = num_obs

        if empty:
            self.A = np.zeros((self.N, self.N))
            self.B = np.zeros((self.N, self.M))
            self.pi = np.zeros((self.N, 1))
        else:
            # Use random initialization
            self.A = np.random.rand(self.N, self.N)
            self.A = self.A / np.sum(self.A, axis=1, keepdims=True)
            self.B = np.random.rand(self.N, self.M)
            self.B = self.B / np.sum(self.B, axis=1, keepdims=True)
            self.pi = np.random.rand(self.N, 1)
            self.pi = self.pi / np.sum(self.pi)


    def forward_one_sequence_with_scale(self, obs_sequence):
        # Number of observations in this sequence
        T = obs_sequence.shape[0]

        # Initialize scaling coefficient
        ct = np.zeros((T, ))

        # Limit the probability to be no lower than thresh to avoid numeric failure
        thresh = 10e-6

        # Initialize forward probabilities
        alpha = np.zeros((self.N, T))

        # Initialize the first step
        Ot = obs_sequence[0]
        alpha[:, [0]] = np.maximum(self.pi * self.B[:, [Ot]], thresh)

        # Scale the first step
        ct[0] = 1 / np.sum(alpha[:, 0])
        alpha[:, 0] = alpha[:, 0] * ct[0]

        # Start propagating forward
        for i in range(1, T):
            # Compute the i th step
            Ot = obs_sequence[i]
            pi = np.dot(alpha[:, [i - 1]].T, self.A).T
            alpha[:, [i]] = np.maximum(pi * self.B[:, [Ot]], thresh)

            # Scale the i th step
            ct[i] = 1 / np.sum(alpha[:, i])
            alpha[:, i] = alpha[:, i] * ct[i]

        log_sequence_prob = - np.sum(np.log(ct))

        return alpha, ct, log_sequence_prob


    def backward_one_sequence_with_scale(self, obs_sequence, ct):
        # Number of observations in this sequence
        T = obs_sequence.shape[0]

        # Initialize backward probabilities
        beta = np.zeros((self.N, T))

        # Initialize the first step
        beta[:, -1] = 1

        # Scale the first step
        beta[:, -1] = beta[:, -1] * ct[-1]

        # Limit the probability to be no lower than thresh to avoid numeric failure
        thresh = 10e-6

        # Start propagating backward
        for i in range(T - 1, 0, -1):
            # Compute the i th step
            Ot = obs_sequence[i]
            beta[:, [i - 1]] = np.maximum(np.dot(self.A, self.B[:, [Ot]] * beta[:, [i]]), thresh)

            # Scale the i th step
            beta[:, i - 1] = beta[:, i - 1] * ct[i - 1]

        return beta


    def reevaluate_model(self, alpha, beta, obs_sequence):
        # The number of time steps
        T = alpha.shape[1]      # Should be the same as beta.shape[1]

        # Initialize xsi
        xsi = np.zeros((self.N, self.N, T - 1))

        # Compute gamma. Equation (27)
        gamma = alpha * beta / np.sum(alpha * beta, axis=0, keepdims=True)

        # Compute xsi for every time instance. Equation (37)
        for i in range(T - 1):
            Ot = obs_sequence[i + 1]
            xsi_numerator = alpha[:, [i]] * self.A * beta[:, [i + 1]].T * self.B[:, [Ot]].T
            xsi_denominator = np.sum(xsi_numerator)

            xsi[:, :, i] = xsi_numerator / xsi_denominator

        # Update the model parameters
        self.pi = gamma[:, [0]]
        self.A = np.sum(xsi, axis=2) / np.sum(gamma[:, :-1], axis=1, keepdims=True)

        gamma_sum_over_t = np.sum(gamma, axis=1, keepdims=True)
        for k in range(self.M):
            mask = obs_sequence[np.newaxis, :] == k
            self.B[:, [k]] = np.sum(gamma * mask, axis=1, keepdims=True) / gamma_sum_over_t


    def baum_welch(self, obs_sequence):
        # Determine the maximum number of iterations and convergence thresh
        max_iter = 1000
        epsilon = 10e-6

        # The first iteration
        alpha, ct, log_sequence_prob = self.forward_one_sequence_with_scale(obs_sequence)
        beta = self.backward_one_sequence_with_scale(obs_sequence, ct)
        self.reevaluate_model(alpha, beta, obs_sequence)

        log_prob_prev = log_sequence_prob

        for i in range(max_iter):
            alpha, ct, log_sequence_prob = self.forward_one_sequence_with_scale(obs_sequence)
            beta = self.backward_one_sequence_with_scale(obs_sequence, ct)
            self.reevaluate_model(alpha, beta, obs_sequence)

            # Display the log probability
            if i % 10 == 0:
                print("Iter:", i, " Log prob:", log_sequence_prob)

            if np.abs(log_sequence_prob - log_prob_prev) < epsilon:
                break

            log_prob_prev = log_sequence_prob

        print(log_sequence_prob)


def train_a_model(name, num_states, num_obs, obs_sequences):
    # Create a model instance
    model = GestureModel(name, num_states, num_obs)

    # Run Baum-Welch on this model
    model.baum_welch(obs_sequences)

    return model


def train_models(folder, num_states, num_obs, classifier=None):
    # Load raw data and classification on training data
    if classifier:
        kmeans, obs_sequences = dl.discretize_raw(folder, num_obs)
    else:   # Use pre-trained classifier to save run time
        kmeans, obs_sequences = dl.discretize_raw(folder, num_obs, classifier)

    # Train a model for each sequence
    one_shot_models = []
    for name, obs_sequence in obs_sequences.items():
        # Create a model for this sequence
        model = GestureModel(name, num_states, num_obs)

        # Feed the observations into the model
        model.baum_welch(obs_sequence)

        one_shot_models.append(model)

    # Integrate these models into 6 models
    gestures = ["beat3", "beat4", "circle", "eight", "inf", "wave"]
    models = []
    counts = []
    for i in range(len(gestures)):
        # Create models with zero parameters
        name = gestures[i]
        model = GestureModel(name, num_states, num_obs, empty=True)
        models.append(model)
        # Initialize counting
        counts.append(0)

    # Loop over the one shot models to put them into these 6 bins
    for model in one_shot_models:
        for i in range(len(gestures)):
            if gestures[i] in model.name:
                models[i].A = models[i].A + model.A
                models[i].B = models[i].B + model.B
                models[i].pi = models[i].pi + model.pi
                counts[i] = counts[i] + 1

                # Stop searching since a model can only belong to one bin
                break

    # Averaging
    for i in range(len(gestures)):
        models[i].A = models[i].A / counts[i]
        models[i].B = models[i].B / counts[i]
        models[i].pi = models[i].pi / counts[i]

    return kmeans, models


def predict(models, obs_sequence):
    # Number of classes
    K = len(models)
    log_probs = np.zeros((K, ))

    # Compute the log probabilities for each model
    for i in range(K):
        _, _, log_prob = models[i].forward_one_sequence_with_scale(obs_sequence)
        log_probs[i] = log_prob

    # Compute the time-normalized probabilities
    probs = np.exp(log_probs / obs_sequence.shape[0])
    probs = probs / np.sum(probs)

    # See if the prediction is confident enough by examining the difference between
    # the top guess and second guess
    top_guess = log_probs[log_probs.argsort()][-1]
    second_guess = log_probs[log_probs.argsort()][-2]

    if top_guess - second_guess < 10:
        label = "Unknown class"
    else:
        number = np.argmax(log_probs)
        label = models[number].name

    # Compute confidence of the prediction
    confidence_naive = (top_guess - second_guess) / (- top_guess)
    P1 = np.exp(top_guess / obs_sequence.shape[0])
    P2 = np.exp(second_guess / obs_sequence.shape[0])
    confidence_timed = (P1 - P2) / P1

    return label, log_probs, probs, confidence_naive, confidence_timed


def plot_correlation(models, show_figure=False):
    # Number of hidden states and observation classes
    N = models[0].A.shape[0]
    M = models[0].B.shape[1]

    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    axs = axs.ravel()
    cmap = cm.get_cmap('hot')
    labels = list(range(1, N + 1))

    for i in range(6):
        name = models[i].name
        A = models[i].A

        cax = axs[i].imshow(A, interpolation="nearest", cmap=cmap)
        axs[i].set_title("Correlation Matrix for " + name)
        axs[i].set_xticklabels(labels)
        axs[i].set_yticklabels(labels)
        ticks = np.linspace(np.min(A), np.max(A), 5)
        fig.colorbar(cax, ax=axs[i], ticks=ticks)

    plt.suptitle("Model Zoo with N = " + str(N) + ", M = " + str(M))

    filename = "Correlation_Matrices"

    plt.savefig(filename)

    if show_figure:
        plt.show()

    plt.close()
    return 0


def plot_probs(name, label, confidence_naive, confidence_timed, log_probs, probs, show_figure=False):
    gestures = [" ","beat3", "beat4", "circle", "eight", "inf", "wave"]
    ind = np.array([0, 1, 2, 3, 4, 5])

    fig = plt.figure(figsize=(12, 5))
    fig.tight_layout()

    ax1 = fig.add_subplot(121)
    ax1.bar(ind, log_probs, facecolor="lightskyblue", align="center")
    ax1.set_xticklabels(gestures)
    for x, y in zip(ind, log_probs):
        ax1.text(x, y, '%.1f' % y, ha='center', va='bottom')
    ax1.set_title("Log likelihoods for each gesture")

    ax2 = fig.add_subplot(122)
    ax2.bar(ind, probs, facecolor="orangered", align="center")
    ax2.set_xticklabels(gestures)
    for x, y in zip(ind, probs):
        ax2.text(x, y, '%.4f' % y, ha='center', va='bottom')
    ax2.set_title("Normalized likelihoods for each gesture")

    confidence_naive = float('%.3f' % confidence_naive)
    confidence_timed = float('%.3f' % confidence_timed)

    plt.suptitle("Sequence name: " + name + "  Predicted Gesture: " + label +
                 "     Confidence 1: " + str(confidence_naive) + "   Confidence 2: " + str(confidence_timed))

    filename = name + "_predictions"

    plt.savefig(filename)

    if show_figure:
        plt.show()

    plt.close()
    return 0