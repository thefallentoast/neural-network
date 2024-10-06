from layer import Layer

class Model(object):
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
    
    def build(self, inputsize):
        for i, layer in enumerate(self.layers):
            output = layer.build(inputsize)

    