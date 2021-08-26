from abc import ABC, abstractmethod
from typing import Iterable, Optional, Union

import numpy as np

class Node(ABC):
    def __init__(self, n, shape) -> None:
        # Checking if either n or shape is provided
        assert (n is not None or shape is not None), "Please provide either number of neurons or the shape of the layer"

        # Initializing number of neurons
        if n is None:
            self.n = np.prod(shape)
        else:
            self.n = n
        
        # Initializing the shape of the layer
        if shape is None:
            self.shape = (n)
        else:
            self.shape = shape

        # Checking if number of neurons and shape are consistent
        assert self.n == np.prod(self.shape), "Number of neurons is inconsistent with the shape of the layer"

    @abstractmethod
    def forward(self) -> None:
        pass

    @abstractmethod
    def activation(self) -> np.ndarray:
        pass
    

class Input(Node):
    
    def __init__(self, n=None, shape=None) -> None:
        # Initializing number of neurons and the shape of the layer
        super().__init__(n=n, shape=shape)

    def forward(self, x: np.ndarray) -> None:
        # Calculate which neurons spike
        self.s = self.activation(x)

    def activation(self, x: np.ndarray) -> np.ndarray:
        # Decides which neurons activate (produce spike)
        return np.where(x != 0, 1, 0)


class LIF(Node):

    def __init__(self, n=None, shape=None, v_rest=-65, v_thresh=-52, decay=0.9, refrac=5) -> None:
        # Initializing number of neurons and the shape of the layer
        super().__init__(n=n, shape=shape)

        # Initializing constants
        self.v_rest = v_rest
        self.v_thresh = v_thresh
        self.decay = decay
        self.refrac = refrac
        
        # Initializing state variables
        self.v = np.full(self.n, v_rest)
        self.s = np.zeros(self.n)
        self.refrac_count = np.zeros(self.n)
        

    def forward(self, x: np.ndarray) -> None:
        # Decaying membrane potential
        self.v = self.decay * (self.v - self.v_rest) + self.v_rest

        # Update refractory period of neurons
        self.refrac_count = np.where(self.refrac_count > 0, self.refrac_count-self.dt, 0)

        # Mask input for neurons in refractory period.
        self.v += np.where(self.refrac_count > 0, 0, x)

        # Calculate which neurons spike
        self.s = self.activation(self.v)

        # Reset membrane potential and refractory period for neurons that spiked
        self.refrac = np.where(self.s, self.refractory_period, self.refrac_count)
        self.v = np.where(self.s, self.v_rest, self.v)


    def activation(self, x: np.ndarray) -> np.ndarray:
        # Decides which neurons activate (produce spike)
        return np.where(x >= self.v_thresh, 1, 0)


class Izhikevich(Node):

    def __init__(self, n=None, shape=None, v_rest=-65, v_thresh=45, excitatory=1) -> None:
        # Initializing number of neurons and the shape of the layer
        super().__init__(n=n, shape=shape)

        # Initializing constants
        self.v_rest = v_rest
        self.v_thresh = v_thresh

        # Determining if neuron will be excitatory and inhibitory 
        if isinstance(excitatory, float):
            if excitatory >= 1:
                ex = np.ones(self.n)  # All neurons are excitatory
            elif excitatory <= 1:
                ex = np.zeros(self.n) # All neurons are inhibitory
            else:
                ex = np
        else:
            ex = np.asarray(excitatory)
        
        assert ex.size == self.n, "Number of neurons in `excitatory` does not match number of neurons in the layer"

        # Initializing state variables
        r = np.random.rand(self.n)
        self.a = np.where(ex > 0, 0.02, 0.02 + 0.08 * r)
        self.b = np.where(ex > 0, 0.2, 0.25 - 0.05 * r)
        self.c = np.where(ex > 0, -65 + 15 * (r**2), -65)
        self.d = np.where(ex > 0, 8 - 6 * (r**2), 2)

        self.v = np.full(self.n, self.v_rest)
        self.u = np.multiply(self.b, self.v)
        self.s = np.zeros(self.n)


    def forward(self, x: np.ndarray) -> None:

        # Update membrane potential and recovery variable
        self.v += self.dt * (0.04 * self.v ** 2 + 5 * self.v + 140 - self.u + x)
        self.u += self.dt * self.a * (self.b * self.v - self.u)

        # Calculate which neurons spike
        self.s = self.activation(self.v)

        # Reset membrane potential and recovery variable for neurons that spiked
        self.v = np.where(self.s, self.c, self.v)
        self.u = np.where(self.s, self.u + self.d, self.u)


    def activation(self, x: np.ndarray) -> np.ndarray:
        # Decides which neurons activate (produce spike)
        return np.where(x >= self.v_thresh, 1, 0)

