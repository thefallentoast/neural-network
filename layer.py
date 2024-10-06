import numpy as np

class Layer(object):
    '''
    base for all other layers.
    
    properties:
        id:
            unique identifier, defaults to 0.
    
    methods:
        call(self, layerInput):
            meant to be defined when inheriting from this class. by default, returns layerInput.
            arguments:
                layerInput (array-like):
                    standard array-like
            returns:
                by default layerInput, but is meant to be redefined when inheriting from this class
    '''
    def __init__(self, identifier=0):
        '''
        Layer object constructor
        arguments:
            identifier
                used in error handling, unique identifier for the 
                layer object.
        '''
        self.id   = identifier
        self.type = 0
    
    def build(self, inputsize):
        '''
        defined when inheriting from layer, e.g making a Pool2D layer.
        takes in inputsize, subscriptable (e.g tuple or np.ndarray.shape)
        outputs outputsize, subscriptable (e.g tuple or np.ndarray.shape)
        '''
        return inputsize
        
    def call(self, layerInput):
        '''
        defined when inheriting from layer, e.g making a Pool2D layer.
        takes in ndarray, outputs ndarray.
        shapes can be user-defined
        '''
        return layerInput
    
    def __call__(self, layerInput):
        try:
            assert hasattr(layerInput, "__array__")
            return self.call(layerInput=layerInput)
        except AssertionError as e:
            return self.call(layerInput=np.array(layerInput))
        except Exception as e:
            raise Warning(f"Exception in layer {self.id}: {e}")
        
class Dense(Layer):
    '''
    Just your regular densely connected NN layer.
    '''
    
    def __init__(self, count, activation="relu"):
        '''
        constructor for the Dense object.
        arguments:
            count:
                neuron count for this layer.
            activation:
                activation names. possible values are (ones that are implemented as of this version):
                    relu
                default:
                    relu
        '''
        self.ncount             = count
        self.neuronvalues       = np.zeros(shape=(self.ncount,))
        
        self.activation         = activation
        
        self.inputsize          = 0 # defined on first run
        self.outputsize         = (self.ncount,) # the output is always as large as the neuron count
        
        self.weights            = np.array([], dtype=np.float64)
        self.biases             = np.array([], dtype=np.float64)
        
    def build(self, inputsize):
        '''
        Randomly sets weights and biases based on inputsize.
        
        Arguments:
            inputsize:
                input count, tuple-like
        
        Returns: 
            outputsize: tuple-like
                the output size or neuron count, can be passed to other Dense layers' build() methods
                in the 1d shape format: (self.ncount,)
        '''
        
        self.weights            = np.random.rand(inputsize[0], self.ncount)
        self.biases             = np.random.rand(self.ncount)
        
        return self.outputsize
        
    def call(self, layerInput):
        try:
            assert layerInput.ndim == 1
        except AssertionError as e:
            raise ValueError("Passed in multidimensional array to a Dense layer. Use Flatten layer to reshape the input.")
        
        self.input = layerInput
        
        self.z = np.dot(self.input, self.weights) + self.biases
        
        return np.array([self.activate(i) for i in self.z])

    def activate(self, x, derivative=False):
        '''
        Pass input through activation function.
        '''
        if derivative:
            if self.activation == "relu":
                return ((abs(x)/x)+1)/2
            
        elif not derivative:
            if self.activation == "relu":
                return (abs(x)+x)/2

    def back(self, gradient, learning_rate):
        '''
        Backpropagates through the layer
        
        Arguments:
            gradient:
                np.array of the gradients of change in cost function relative to this layer's activations
        Returns:
            np.array of gradients representing the gradient of the cost with respect to each input
        '''
        
        gradient_activation = np.array([self.activate(i, derivative=True) for i in self.z])

        gradient_z = gradient_activation * gradient

        gradient_weights = np.dot(self.input[:, np.newaxis], gradient_z[np.newaxis, :])
        gradient_biases = gradient_z

        gradient_input = np.dot(self.weights, gradient_z)

        self.weights = self.weights - (learning_rate * gradient_weights)
        self.biases = self.biases - (learning_rate * gradient_biases)

        return gradient_input, learning_rate

class Dropout(Layer):
    '''
    Randomly sets values to 0
    '''

    def __init__(self, dropout_chance, random_state=1337):
        '''
        Constructor for the Dropout layer.
        
        Arguments:
            dropout_chance:
                The chance for a value to be set to 0
            random_state:
                The seed used internally

        '''

        self.chance = dropout_chance

    def build(self, inputsize):
        return inputsize

    def back(self, gradient, learning_rate):
        return gradient, learning_rate
    
    def call(self, layerInput):
        indices = np.random.choice(
            layerInput.shape[1] * layerInput.shape[0],
            replace=False, 
            size=int(
                layerInput.shape[1] * layerInput.shape[0]*0.2
                )
            )
        self.output = layerInput
        self.output[np.unravel_index(indices, my_array.shape)] = 0 
        return self.output