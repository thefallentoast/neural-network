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
        self.id = identifier
    
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
        self.ncount = count
        self.neuronvalues = np.zeros(shape=(self.ncount,))
        
        self.activation = activation
        
        self.inputsize = 0 # defined on first run
        self.outputsize = (self.ncount,) # the output is always as large as the neuron count
        
        self.weights = np.array([], dtype=np.float64)
        self.biases = np.array([], dtype=np.float64)
        
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
        
        self.weights = np.random.rand(inputsize[0], self.ncount)
        self.biases = np.random.rand(self.ncount)
        
        return self.outputsize
        
    def call(self, layerInput):
        try:
            assert layerInput.ndim == 1
        except AssertionError as e:
            raise ValueError("Passed in multidimensional array to a Dense layer. Use Flatten layer to reshape the input.")
        
        self.input = layerInput
        
        self.z = np.dot(self.input, self.weights) + self.biases
        
        return self.activate(self.z)

    def activate(self, input, derivative=False):
        '''
        Pass input through activation function.
        '''
        if derivative:
            if self.activation == "relu":
                return 1 if input > 0 else 0
            
        elif not derivative:
            if self.activation == "relu":
                return input if input > 0 else 0

    def back(self, gradient):
        '''
        Backpropagates through the layer
        
        Arguments:
            gradient:
                Rate of change in cost function relative to the activation.
                Example: 1 output neuron, squared error
                error = (output - correct) ^ 2
                gradient (the derivative) = 2 * (output - correct)
        Returns:
            np.array of gradients representing the gradient of the cost with respect to each input
        '''
        
        # the total gradient of change for one weight w connecting the ith neuron of the L layer to the jth neuron of the
        # L-1 layer is:
        # deltacost/deltaw = deltacost/deltaactivation * deltaactivation/deltaz * deltaz/deltaw
        # resolving for:
        #     deltaz/deltaw:
        #         w influences z by the previous activation, so the previous activation's influence here is w
        #         thus, deltaz/deltaw = a
        #     deltaactivation/deltaz:
        #         z influences a by the derivative of the activation, thus
        #         deltaactivation/deltaz = self.activate(z, derivative=True)
        #     deltacost/deltaactivation:
        #         a influences C by the derivative of the cost funcion, passed in by argument, thus
        #         deltacost/deltaactivation = gradient
        
        