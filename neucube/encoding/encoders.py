""" Encoders """
# Authors: Sugam Budhraja <sugam11nov@gmail.com>
# License: Open

import numpy as np
from scipy.signal import firwin
from scipy.ndimage import uniform_filter1d

from abc import ABC, abstractmethod


class Encoder(ABC):

    def __init__(self) -> None:
        """Initializes default parameter values for the encoder.

        Returns
        -------
        None
        """
        super().__init__()

    def encode_dataset(self, data: np.ndarray, reconstruct: bool = False) -> np.ndarray:
        """Converts given dataset to spike trains.

        Parameters
        ----------
        data : array-like of shape (n_samples, n_timesteps, n_features)
            Dataset to be converted to spike trains.

        reconstruct: bool, default=False
            If true, the function also returns the reconstructed dataset
            from the spike trains.

        Returns
        -------
        spikes : array-like of shape (n_samples, n_timesteps, n_features)
            Spike trains for given dataset.
        """

        encoded_data = np.zeros(data.shape)
        if reconstruct:
            reconstructed_data = np.zeros(data.shape)

        for sample in range(data.shape[0]):
            for feature in range(data.shape[2]):
                encoded_data[sample, :, feature] = self.encode(
                    data[sample, :, feature])
                if reconstruct:
                    reconstructed_data[sample, :, feature] = self.decode(
                        data[sample, :, feature])

        if reconstruct:
            return encoded_data, reconstructed_data
        return encoded_data

    @abstractmethod
    def encode(self, signal: np.ndarray) -> np.ndarray:
        """Converts 1d timeseries to spike train.

        Parameters
        ----------
        signal : array-like of shape (n_timesteps,)
            Signal to be converted to spike train.

        Returns
        -------
        spikes : array-like of shape (n_timesteps,)
            Encoded spike train from signal.
        """
        pass

    @abstractmethod
    def decode(self, spikes: np.ndarray) -> np.ndarray:
        """Reconstructs 1d timeseries from spike train.

        Parameters
        ----------
        spikes : array-like of shape (n_timesteps,)
            Spike train to converted to signal.

        Returns
        -------
        signal : array-like of shape (n_timesteps,)
            Signal reconstructed from spike train.
        """
        pass


class StepForward(Encoder):

    """Step Forward encoding algorithm to convert timeseries
     data to spike train.

    Parameters
    ----------
    threshold : float, default=0.3
        Threshold value used to determine if a spike occurs 
        or not.

    initial_val : float, default=0
        First value of signal to be encoded/decoded. 

    Examples
    --------
    >>> signal = [0.2, 0.5, 0.1, 0.3, 0.7]
    >>> from neucube.encoding import StepForward
    >>> enc_sf = StepForward(threshold=0.3)
    >>> spikes = enc_sf.encode(signal)
    >>> print(spikes)
    [0, 1, -1, 0, 1]
    >>> reconstructed_signal = enc_sf.decode(spikes)
    >>> print(reconstructed_signal)
    [0.2, 0.5, 0.2, 0.2, 0.5]

    Notes
    -----

    """

    def __init__(self, threshold: float = 0.3, initial_val: float = 0.0) -> None:
        """Initializes default parameter values for the encoder.

        Parameters
        ----------
        threshold: float, default=0.3
            The default value for the threshold parameter used 
            to determine whether a spike is encoded.

        initial_val: float, default=0.0
            The default value of initial_val used for decoding 
            the spike train.

        Returns
        -------
        None
        """
        self.threshold = threshold
        self.initial_val = initial_val

    def encode(self, signal: np.ndarray, threshold: float = None) -> np.ndarray:
        """Converts 1d timeseries to spike train.

        Parameters
        ----------
        signal : array-like of shape (n_timesteps,)
            Signal to be converted to spike train.

        threshold: float, default=None
            The value for the threshold parameter that decides
            whether spike is encoded. If `None` then class 
            attribute `threshold` is used.

        Returns
        -------
        spikes : array-like of shape (n_timesteps,)
            Encoded spike train from signal.
        """

        # Initializing the spike train to be an array of zeros with same number of timesteps as the signal.
        spikes = np.zeros(signal.shape, dtype=np.int8)

        # If threshold value is passed to the function, it is used. Otherwise threshold value passed during initalization is used.
        if threshold == None:
            threshold = self.threshold

        # Saving the first value of the signal for decode function.
        self.initial_val = signal[0].copy()

        # Initializing base to first value of the signal. .copy() used to avoid affecting the input signal.
        base = signal[0].copy()

        # Iterating over all timesteps
        for t in range(1, signal.size):

            # If signal value at time step t is greater than base+threshold, a positive spike is encoded.
            if signal[t] >= base + threshold:
                spikes[t] = 1
                base += threshold

            # If signal value at time step t is less than base-threshold, a negative spike is encoded.
            elif signal[t] <= base - threshold:
                spikes[t] = -1
                base -= threshold

        # Returns spike train
        return spikes

    def decode(self, spikes: np.ndarray, threshold: float = None, initial_val: float = None) -> np.ndarray:
        """Reconstructs 1d timeseries from spike train.

        Parameters
        ----------
        spikes : array-like of shape (n_timesteps,)
            Spike train to converted to signal.

        threshold: float, default=None
            The value for the threshold parameter that decides
            whether spike is encoded. If `None` then class 
            attribute `threshold` is used.

        initial_val: float, default=0.0
            The first value of `signal` that is used to decode 
            the spike train.

        Returns
        -------
        signal : array-like of shape (n_timesteps,)
            Signal reconstructed from spike train.
        """

        # If threshold value is passed to the function, it is used. Otherwise threshold value passed during initalization is used.
        if threshold == None:
            threshold = self.threshold

        # If initial_val value is passed to the function, it is used. Otherwise initial_val set by encode() or passed during initalization is used.
        if initial_val == None:
            initial_val = self.initial_val

        # Signal is initialized to array filled with initial_val. Cumulative sum of spikes * threshold is then added to it which represents change in the signal over time.
        signal = np.full(spikes.shape, initial_val) + \
            np.cumsum(spikes)*threshold

        # Returns signal
        return signal


class BensSpikerAlgorithm(Encoder):

    """Ben's Spiker Algorithm to convert timeseries
     data to spike train.

    Parameters
    ----------

    filter: array-like of shape (n_filter,)
        Filter that is used to model the signal.

    window_size: int, default=20
        Size of filter. Parameter is passed to `firwin()`.

    cutoff_freq: float, default=0.1
        Cutoff frequency defines the maximum value of
        the filter. Parameter is passed to `firwin()`.
        
    threshold : float, default=0.3
        Threshold value used to determine if a spike occurs 
        or not.

    initial_val : float, default=0
        First value of signal to be encoded/decoded.

    Examples
    --------
    >>> signal = [0.2, 0.5, 0.1, 0.3, 0.7]
    >>> from neucube.encoding import StepForward
    >>> enc_sf = StepForward(threshold=0.3)
    >>> spikes = enc_sf.encode(signal)
    >>> print(spikes)
    [0, 1, -1, 0, 1]
    >>> reconstructed_signal = enc_sf.decode(spikes)
    >>> print(reconstructed_signal)
    [0.2, 0.5, 0.2, 0.2, 0.5]

    Notes
    -----

    """

    def __init__(self, threshold: float = 0.955, filter: np.ndarray = None, window_size: int = 20, cutoff_freq: float = 0.1, initial_val: float = 0.0) -> None:
        """Initializes default parameter values for the encoder.

        Parameters
        ----------
        
        filter: array-like of shape (n_filter,)
            The default value of `filter` that is used to 
            model the signal. If not specified, it is calculated
            by passing `window_size` and `cutoff_freq` to `firwin()`.

        window_size: int, default=20
            Size of filter. Parameter is passed to `firwin()`.

        cutoff_freq: float, default=0.1
            Cutoff frequency defines the maximum value of
            the filter. Parameter is passed to `firwin()`.

        threshold: float, default=0.3
            The default value for the threshold parameter used in the
            encoding algorithm.

        initial_val: float, default=0.0
            The default value of initial_val used for decoding 
            the spike train.

        Returns
        -------
        None
        """
        self.threshold = threshold
        self.window_size = window_size
        self.cutoff_freq = cutoff_freq
        if filter:  # If filter is passed, use it
            self.filter = filter
        else:  # Else use win_size and cutoff_freq to construct filter
            self.filter = firwin(self.window_size, self.cutoff_freq)
        self.initial_val = initial_val

    def encode(self, signal: np.ndarray, filter: np.ndarray = None, threshold: float = None) -> np.ndarray:
        """Converts 1d timeseries to spike train.

        Parameters
        ----------
        signal: array-like of shape (n_timesteps,)
            Signal to be converted to spike train.

        filter: array-like of shape (n_filter,)
            Filter that is used to model the signal.
            
        threshold: float, default=None
            The value for the threshold parameter that decides
            whether spike is encoded. If `None` then class 
            attribute `threshold` is used.

        Returns
        -------
        spikes : array-like of shape (n_timesteps,)
            Encoded spike train from signal.
        """

        assert (signal >= 0).all(), "Inputs must be non-negative"

        # Initializing the spike train to be an array of zeros with same number of timesteps as the signal.
        spikes = np.zeros(signal.shape, dtype=np.int8)

        # If filter value is passed to the function, it is used. Otherwise filter value passed during initalization is used.
        if filter == None:
            filter = self.filter

        # If threshold value is passed to the function, it is used. Otherwise threshold value passed during initalization is used.
        if threshold == None:
            threshold = self.threshold

        # Saving the first value of the signal for decode function.
        self.initial_val = signal[0].copy()

        # Iterating over all timesteps
        for i in range(signal.size):

            # `j` accounts for how much of filter is used. Near end of signal, part of filter is used
            j = min(filter.size, signal.size - i)

            # Checking if subtracting filter from signal is possible.
            error1 = np.sum(np.abs(signal[i:i+j] - filter[:j]))
            error2 = np.sum(np.abs(signal[i:i+j]))

            # If it is possible to subtract filter, a spike is encoded and the filter is subtracted.
            if error1 <= (error2 - threshold):
                spikes[i] = 1
                signal[i:i+j] -= filter[:j]

        # Returns spike train.
        return spikes

    def decode(self, spikes: np.ndarray, filter: np.ndarray = None, threshold: float = None, initial_val: float = None) -> np.ndarray:
        """Reconstructs 1d timeseries from spike train.

        Parameters
        ----------
        spikes : array-like of shape (n_timesteps,)
            Spike train to converted to signal.

        filter: array-like of shape (n_filter,)
            Filter that is used to model the signal.

        threshold: float, default=None
            The value for the threshold parameter that decides
            whether spike is encoded. If `None` then class 
            attribute `threshold` is used.

        initial_val: float, default=0.0
            The first value of `signal` that is used to decode 
            the spike train.

        Returns
        -------
        signal : array-like of shape (n_timesteps,)
            Signal reconstructed from spike train.

        """

        # If filter is passed to the function, it is used. Otherwise filled passed during initalization is used.
        if filter == None:
            filter = self.filter

        # If threshold value is passed to the function, it is used. Otherwise threshold value passed during initalization is used.
        if threshold == None:
            threshold = self.threshold

        # If initial_val value is passed to the function, it is used. Otherwise initial_val set by encode() or passed during initalization is used.
        if initial_val == None:
            initial_val = self.initial_val

        # Signal is initialized to array filled with initial_val.
        signal = np.full(spikes.size, initial_val)

        # Whenever we encounter a spike, the filter is added to the signal starting from that timestep.
        for i in np.where(spikes == 1)[0]:
            j = min(filter.size, signal.size - i)
            signal[i:i+j] += filter[:j]

        # Returns signal
        return signal


class ThresholdBasedRepresentation(Encoder):

    """Threshold Based Representation algorithm to convert timeseries
     data to spike train.

    Parameters
    ----------
    factor: float, default=None
        Factor parameter is used to calculate how much
        standard deviation contributes to `threshold`.
        
    threshold : float, default=0.3
        Threshold value that decides whether a spike is encoded.

    initial_val : float, default=0
        First value of signal to be encoded/decoded.

    Examples
    --------
    >>> signal = [0.2, 0.5, 0.1, 0.3, 0.7]
    >>> from encoder import StepForward
    >>> m_sf = StepForward(threshold=3)
    >>> spikes = m_sf.encode(signal)
    >>> print(spikes)
    [0, 1, -1, 0, 1]
    >>> print(m_sf.decode(spikes))
    [0.2, 0.5, 0.2, 0.2, 0.5]

    Notes
    -----

    """

    def __init__(self, factor: float = 0.5, threshold: float = 0.3, initial_val: float = 0.0) -> None:
        """Initializes default parameter values for the encoder.

        Parameters
        ----------
        factor: float, default=None
            The default value for the `factor` parameter that will be used
            to calculate `threshold`.

        threshold: float, default=0.3
            The default value for the threshold parameter used in the
            encoding algorithm.

        initial_val: float, default=0.0
            The default value of initial_val used for decoding 
            the spike train.

        Returns
        -------
        None
        """
        self.factor = factor
        self.threshold = threshold
        self.initial_val = initial_val

    def encode(self, signal: np.ndarray, factor: float = None) -> np.ndarray:
        """Converts 1d timeseries to spike train.

        Parameters
        ----------
        signal : array-like of shape (n_timesteps,)
            Signal to be converted to spike train.
            
        factor: float, default=None
            The value for the factor parameter that is used
            to calculate `threshold`. If `None` then class 
            attribute `factor` is used.

        Returns
        -------
        spikes : array-like of shape (n_timesteps,)
            Encoded spike train from signal.
        """

        # Initializing the spike train to be an array of zeros with same number of timesteps as the signal.
        spikes = np.zeros(signal.shape, dtype=np.int8)

        # If factor is passed to the function, it is used. Otherwise factor value passed during initalization is used.
        if factor == None:
            factor = self.factor

        # diff stores difference between consecutive elements.
        diff = signal[1:] - signal[:-1]
        diff = np.concatenate(([0], diff))

        # Threshold is set to mean of diff array + factor * standard deviation of diff array.
        threshold = diff.mean() + factor * diff.std()

        # Saving the threshold value for decode function.
        self.threshold = threshold
        # Saving the first value of the signal for decode function.
        self.initial_val = signal[0]

        # Wherever difference between consecutive elements is greater than threshold, a spike is encoded.
        spikes = np.where(diff > threshold, 1., spikes)
        spikes = np.where(diff < -threshold, -1., spikes)

        # Returns spike train.
        return spikes

    def decode(self, spikes: np.ndarray, threshold: float = None, initial_val: float = None) -> np.ndarray:
        """Reconstructs 1d timeseries from spike train.

        Parameters
        ----------
        spikes : array-like of shape (n_timesteps,)
            Spike train to converted to signal.
            
        threshold: float, default=None
            The value for the threshold parameter that decides
            whether spike is encoded. If `None` then class 
            attribute `threshold` is used.

        initial_val: float, default=0.0
            The first value of `signal` that is used to decode 
            the spike train.

        Returns
        -------
        signal : array-like of shape (n_timesteps,)
            Signal reconstructed from spike train.
        """

        # If threshold value is passed to the function, it is used. Otherwise threshold value passed during initalization is used.
        if threshold == None:
            threshold = self.threshold

        # If initial_val value is passed to the function, it is used. Otherwise initial_val set by encode() or passed during initalization is used.
        if initial_val == None:
            initial_val = self.initial_val

        # Signal is initialized to array filled with initial_val. Cumulative sum of spikes * threshold is then added to it which represents change in the signal over time.
        signal = np.full(spikes.shape, initial_val) + \
            np.cumsum(spikes)*threshold

        # Returns signal.
        return signal


class MovingWindow(Encoder):

    """Moving Window algorithm to convert timeseries
     data to spike train..

    Parameters
    ----------
    window: int, default=5
        Size of sliding window.

    threshold : float, default=0.3
        Threshold value that decides whether a spike is encoded.

    initial_val : float, default=0
        First value of signal to be encoded/decoded.

    Examples
    --------
    >>> signal = [0.2, 0.5, 0.1, 0.3, 0.7]
    >>> from encoder import StepForward
    >>> m_sf = StepForward(threshold=3)
    >>> spikes = m_sf.encode(signal)
    >>> print(spikes)
    [0, 1, -1, 0, 1]
    >>> print(m_sf.decode(spikes))
    [0.2, 0.5, 0.2, 0.2, 0.5]

    Notes
    -----

    """

    def __init__(self, window: int = 5, threshold: float = 0.3, initial_val: float = 0.0) -> None:
        """Initializes default parameter values for the encoder.

        Parameters
        ----------
        window: int, default=5
            Size of sliding window.

        threshold: float, default=0.3
            The default value for the threshold parameter used in the
            encoding algorithm.

        initial_val: float, default=0.0
            The default value of initial_val used for decoding 
            the spike train.

        Returns
        -------
        None
        """
        self.window = window
        self.threshold = threshold
        self.initial_val = initial_val

    def encode(self, signal: np.ndarray, window: int = None, threshold: float = None) -> np.ndarray:
        """Converts 1d timeseries to spike train.

        Parameters
        ----------
        signal : array-like of shape (n_timesteps,)
            Signal to be converted to spike train.

        window: int, default=5
            Size of sliding window.

        threshold: float, default=None
            The value for the threshold parameter that decides
            whether spike is encoded. If `None` then class 
            attribute `threshold` is used.

        Returns
        -------
        spikes : array-like of shape (n_timesteps,)
            Encoded spike train from signal.
        """

        # Initializing the spike train to be an array of zeros with same number of timesteps as the signal.
        spikes = np.zeros(signal.shape, dtype=np.int8)

        # If window value is passed to the function, it is used. Otherwise window value passed during initalization is used.
        if window == None:
            window = self.window

        # If threshold value is passed to the function, it is used. Otherwise threshold value passed during initalization is used.
        if threshold == None:
            threshold = self.threshold

        # Saving the first value of the `signal` for `decode()`.
        self.initial_val = signal[0]

        # `base` stores sliding window mean
        rolling_mean = uniform_filter1d(signal, window)[
            (window//2):signal.size-((window-1)//2)]
        base = np.concatenate(
            (np.full(window-1, rolling_mean[0]), rolling_mean))

        # Wherever difference between signal and base is more than threshold, spike is encoded.
        spikes = np.where(signal > base+threshold, 1, spikes)
        spikes = np.where(signal < base-threshold, -1, spikes)

        # Returns spike train.
        return spikes

    def decode(self, spikes: np.ndarray, threshold: float = None, initial_val: float = None) -> np.ndarray:
        """Reconstructs 1d timeseries from spike train.

        Parameters
        ----------
        spikes : array-like of shape (n_timesteps,)
            Spike train to converted to signal.

        threshold: float, default=None
            The value for the threshold parameter that decides
            whether spike is encoded. If `None` then class 
            attribute `threshold` is used.

        initial_val: float, default=0.0
            The first value of `signal` that is used to decode 
            the spike train.

        Returns
        -------
        signal : array-like of shape (n_timesteps,)
            Signal reconstructed from spike train.
        """

        # If threshold value is passed to the function, it is used. Otherwise threshold value passed during initalization is used.
        if threshold == None:
            threshold = self.threshold

        # If initial_val value is passed to the function, it is used. Otherwise initial_val set by encode() or passed during initalization is used.
        if initial_val == None:
            initial_val = self.initial_val

        # Signal is initialized to array filled with initial_val. Cumulative sum of spikes * threshold is then added to it which represents change in the signal over time.
        signal = np.full(spikes.shape, initial_val) + \
            np.cumsum(spikes)*threshold

        # Returns signal.
        return signal
