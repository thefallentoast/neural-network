import numpy as np
from .layer import Layer


class NeuralNetwork(object):
    def __init__(self):
        self.layers = []
        
    def add(self, layer):
        '''
        appends Layer object to end of network.
        arguments:
            layer
                either Layer or Layer-like object
        returns:
            None
        raises:
            TypeError if layer argument is not Layer subclass or Layer
        '''
        try:
            assert issubclass(layer, Layer) or isinstance(layer, Layer)
            self.layers.append(layer)
        except AssertionError as e:
            raise Warning("Attempted to add object which is not layer to layers of NeuralNetwork object.")
    
    def remove(self, index):
        pass