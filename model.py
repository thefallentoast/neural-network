from layer import Layer
from mathutils import round_fixed

import random
import numpy as np



class Model(object):
    '''
    A model that works much like keras' Sequential model.
    '''
    def __init__(self, layers=[]):
        '''
        Model() constructor.
        Arguments:
            layers:
                (optional) list of layers to add to the model
        '''

        if isinstance(layers, (list, tuple)):
            self.layers = layers
        else:
            self.layers = [layers]

        self.losses = {
            "mean_squared_error": self.MSE
        }
        
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
            assert issubclass(type(layer), Layer) or isinstance(layer, Layer)
            self.layers.append(layer)
        except AssertionError as e:
            raise TypeError("Attempted to add object which is not layer to layers of Model object.")
    
    def compile(self, learning_rate, loss, input_shape):
        '''
        Compiles the model based on parameters.
        Parameters (passed in through dict):
            learning_rate:
                The learning rate used in backpropagation.
            loss:
                The loss to use.
            input_shape:
                The shape of the input to the model.
        '''
        
        self.lr            = learning_rate
        self.loss          = loss
        self.input_shape   = input_shape
        

        shape = self.input_shape
        for layer in self.layers:
            shape = layer.build(shape)

        self.output_shape = shape


    def runLayersOn(self, model_input):
        output = model_input
        for i, layer in enumerate(self.layers):
            output = layer(output)
        self.output = output
        return output

    def trainLayers(self, error_gradient):
        gradient = error_gradient
        for i, layer in enumerate(self.layers[::-1]):
            gradient = layer.train(gradient, self.lr)

    def calculateError(self, output, expected, gradient=False):
        return self.losses[self.loss](output, expected, gradient=gradient)

    # the errors used by self.calculateError
    def MSE(self, output, expected, gradient=False):
        if gradient:
            return (2 / output.shape[0]) * (output - expected)
        else:
            return np.mean((output - expected) ** 2)

    def fit(self, Xtr, ytr, epochs=1000, batch_size=32):
        '''
        fits the model.
        '''

        X_batches = [] # I'll use a list because I need batches.append()
        y_batches = [] # Same here

        # first, we split the batches
        
        for bid in range( round_fixed(len(Xtr) / batch_size) ):
            X_batches.append(Xtr[bid * batch_size : (bid+1) * batch_size])
            y_batches.append(ytr[bid * batch_size : (bid+1) * batch_size])

        print(X_batches)
        
        for epoch in range(epochs):
            batch_index = random.choice(list(enumerate(X_batches)))[0]  # Picks a random index, since I need to use it on X and y
            # note 1 about the above line: enumerate() doesn't return an iterable, and has no len(), so you cast it to list()
            X = X_batches[batch_index]
            y = y_batches[batch_index]

            for layerInput, expectedOutput in zip(X, y): # for simplicity
                layerOutput             = self.runLayersOn(layerInput)

                error                   = self.calculateError(layerOutput, expectedOutput, gradient=False)
                error_gradient          = self.calculateError(layerOutput, expectedOutput, gradient=True)

                self.trainLayers(error_gradient=error_gradient)

            if epoch % 10 == 0:
                print(f"Epoch {epoch} / {epochs}: loss: {error}")
    def __call__(self, model_input):
        return self.runLayersOn(model_input)