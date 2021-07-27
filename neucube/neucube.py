""" NeuCube """
# Authors: Sugam Budhraja <sugam11nov@gmail.com>
# License: Open

import numpy as np
import scipy as sp
import pandas as pd
import os


class NeuCube():

    """The NeuCube architecture.
    Read more in the paper .

    Parameters
    ----------
    dt : float, default=1
        Time-step for simulating the spiking neurons (in milliseconds). 
        By default set to 1ms.


    electrodes : list, default=None
        Names of input EEG electrodes.

    coordinates : {'talairach'} or custom, default='talairach'
        weight function used in prediction.  Possible values:
        - 'talairach' : Using talairach brain atlas to initialize 
        1471 reservoir neurons
        - [custom] : user-defined coordinates for reservoir neurons

    seed : int, default=42
        Setting seed to random function for reproducability.

    Examples
    --------
    >>> X = [[0.213, 0.312], [0.565, 0.342], [0.288, 0.944], [0.763, 0.325]]
    >>> y = [0, 0, 1, 1]
    >>> from neucube import NeuCube
    >>> m_neu = NeuCube(electrodes=['F1', 'F2'])
    >>> m_neu.fit(X, y)
    NeuCube(...)
    >>> print(m_neu.predict([[0.212, 0.342]]))
    [0]

    Notes
    -----

    """

    def __init__(self, electrodes=None, coordinates='talairach', mapping='koessler', learning_method='STDP', seed=42, device='cpu'):
        self.electrodes = electrodes
        self.coordinates = coordinates
        self.mapping = mapping
        self.learning_method = learning_method
        np.random.seed(seed)
        self.device = device
        self._initialize_structure()

    def _initialize_structure(self):
        print(os.getcwd())
        # Initializing neuron positions
        if self.coordinates.lower() == 'talairach':
            self.reservoir_neuron_positions = pd.read_csv(
                'assets/talairach_brain_coordinates.csv', header=None).values
        else:
            self.reservoir_neuron_positions = self.coordinates
        # Number of reservoir neurons
        self.n_reservoir = self.reservoir_neuron_positions.shape[0]

        # Initializing input electrode positions
        if self.mapping == 'koessler':
            if (self.electrodes == None).all():
                raise ValueError('Electrode names not provided')

            koessler_mapping = pd.read_csv(
                'assets/koessler_mapping.csv', header=None, index_col=0).T

            if not (np.isin(self.electrodes, koessler_mapping.columns).all()):
                raise ValueError('Some electrode names not identified')

            self.input_neuron_positions = koessler_mapping[self.electrodes].T.values
        else:
            self.input_neuron_positions = self.mapping
        # Number of input neurons
        self.n_input = self.input_neuron_positions.shape[0]

        # Calculating distance matrix between every pair of reservoir neuron
        reservoir_dist = sp.spatial.distance.squareform(sp.spatial.distance.pdist(
            self.reservoir_neuron_positions))
        # Normalizing distance
        reservoir_dist /= np.max(reservoir_dist)

        # # Calculating distance between input and reservoir neurons
        input_dist = sp.spatial.distance.cdist(
            self.input_neuron_positions, self.reservoir_neuron_positions)
        # Normalizing distance (delibarately using same max dist for normalization)
        input_dist /= np.max(reservoir_dist)

        # Initializing reservoir connections based on small world connectivity
        conn_prob = 0.15
        small_world_conn = 2.5

        # Calculating connection probabilities
        reservoir_prob = conn_prob * \
            np.exp(-(reservoir_dist/small_world_conn)**2)
        input_prob = conn_prob * \
            np.exp(-(input_dist/small_world_conn)**2)

        # Initializing random weight for connections
        self.reservoir_weight_matrix = np.random.rand(
            self.n_reservoir, self.n_reservoir)
        self.input_weight_matrix = np.random.rand(
            self.n_input, self.n_reservoir)

        # Determinig which connections will exist and alloting them their initial weight
        self.reservoir_weight_matrix = np.where(
            self.reservoir_weight_matrix < reservoir_prob, self.reservoir_weight_matrix*10, 0)
        self.input_weight_matrix = np.where(
            self.input_weight_matrix < input_prob, self.input_weight_matrix*10, 0)

        # Computing adjacency matrix and adjacency list for faster calculation during training
        self.reservoir_connections_matrix = self.reservoir_weight_matrix > 0
        reservoir_connections = np.where(self.reservoir_connections_matrix)
        self.reservoir_connections_list = np.split(reservoir_connections[1], np.unique(
            reservoir_connections[0], return_index=True)[1][1:])

        self.input_connections_matrix = self.input_weight_matrix > 0
        input_connections = np.where(self.input_connections_matrix)
        self.input_connections_list = np.split(input_connections[1], np.unique(
            input_connections[0], return_index=True)[1][1:])

    def fit(self, X):
        n_samples = X.shape[0]  # Number of samples in training data
        n_timesteps = X.shape[1]  # Number of timesteps in one training sample
        dt = 1  # Time step in ms
        v_rest = -65  # Resting neuron membrane potential
        v_thresh = -52  # Spiking threshold. if membrane potential crosses this, neuron spikes
        refrac_period = 5  # Neuron refractory period in ms
        stdp_rate = 0.1  # STDP rate is maximum change in weight of a connection at one timestep
        tau = 1  # STDP time constant
        tc_decay = 100  # LIF Decay time constant
        decay = np.exp(-dt/tc_decay)
        # Number of times each neuron spikes. To be used for classification.
        self.spike_count = np.zeros((n_samples, self.n_reservoir))

        for sample in range(n_samples):
            # Current membrane potential for all reservoir neurons. Initialized to v_rest
            v = np.full(self.n_reservoir, v_rest)
            # Time left in refractory period. Initialized to 0
            refrac = np.zeros(self.n_reservoir)
            # Stores whether a neuron spiked at current timestep
            s = np.zeros(self.n_reservoir, dtype=bool)
            # Time at which neuron last spiked. Initialized to 0
            last_s = np.zeros(self.n_reservoir)
            # Spike trains of input neurons
            input_s = X[sample].astype(bool)

            for t in range(n_timesteps):

                # dv = np.sum(
                #     self.input_weight_matrix[input_s[t]], axis=0)
                # dv += np.sum(
                #     self.reservoir_weight_matrix[s], axis=0) # np.sum 2x slower than + when summing along axis

                # Change in membrane potential for all neurons
                dv = np.zeros(self.n_reservoir)
                for neuron in np.where(input_s[t])[0]:
                    # dv because of input spikes
                    dv += self.input_weight_matrix[neuron]
                for neuron in np.where(s)[0]:
                    # dv because of reservoir spikes
                    dv += self.reservoir_weight_matrix[neuron]
                # Membrane potential of neurons in refrac period doesn't change
                dv = np.where(refrac > 0, 0, dv)

                # Decaying membrane potential
                v = decay * (v - v_rest) + v_rest
                v += dv  # Updating membrane potential
                # Updating refrac count
                refrac = np.where(refrac > 0, refrac-dt, 0)
                s = v >= v_thresh  # calculating which neurons spike at current timestep
                # Updating refrac of neurons that spiked
                refrac = np.where(s, refrac_period, refrac)
                # Updating membrane potential of neurons that spiked
                v = np.where(s, v_rest, v)

                # STDP
                dw = stdp_rate * np.exp(-(t - last_s + 1)/tau)
                for pre in np.where(s)[0]:
                    self.reservoir_weight_matrix[pre] -= np.where(
                        self.reservoir_weight_matrix[pre] > 0, dw, 0)
                for post in np.where(s)[0]:
                    self.reservoir_weight_matrix[:, post] += np.where(
                        self.reservoir_weight_matrix[:, post] > 0, dw, 0)
                # Updating last spike timestep for neurons that spiked at this timestep
                last_s = np.where(s, t, last_s)
                # Updating spike count feature vector
                self.spike_count[sample] += s

        return self

    def predict(self, X, y):
        predictions = []
        return predictions
