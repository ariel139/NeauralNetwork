import numpy as np
from enum import Enum
from softmax import softmax
from Layer_type import Layer_type
from errors import Error
from RectifiedLinearUnit import Relu

err = Error()


class object_type(Enum):
    weight = 1
    neuron = 2
    bias = 3


class Layer:

    def __init__(self,input_size:int,  output_size: int,
                 layer_type: Layer_type, input_data = None):
        """
            constructor
            :param input_data the input data fot the layer
            :param output_size the size of the vector of the output
            :param layer_type the type of the layer defined by the Enum Layer_Type
            :raises when got none valid data type for one of the parameters or none valid shape of input data
            """
        # Error handling

        err.type_error(output_size, int)
        err.type_error(layer_type, Layer_type)
        if input_data is not None:
            err.type_error(input_data, np.ndarray)
            if input_data.ndim != 1:
                raise Exception(
                  f'input data must be one-dimensional got: {input_data.ndim} dimensions'
                )

        # setting the properties
        self.layer_type = layer_type
        self.input_data = input_data
        self.input_size = input_size
        self.out_put = None
        self.weights = np.random.rand(input_size, output_size)
        self.bias = np.random.rand(output_size)
        self.partial_gradient = None

    def forward(self, activation=Relu.regular):
        """
            does the forward pass for the network.
            :param activation default RELU for best results
            :returns numpy array with the result of the layer
            :raise in case of none valid input
            """
        # error handle
        # err.type_error(activation, callable)

        # calculating the output
        output = np.dot(self.input_data, self.weights)
        # TODO: add bias do in more advenced level
        func = activation

        # pass throw activation
        if self.layer_type == Layer_type.Output:
            softmx = softmax(output)
            softmx.regular(100000)
            self.out_put = softmx.output_vector
        else:
            if self.layer_type == Layer_type.Hidden and activation != Relu.regular:
                vectorised = np.vectorize(func)
                self.out_put = vectorised(output)
            else:
                relu = Relu(output)
                relu.regular()
                self.out_put = relu.output_vector

    def backward(self, y: np.ndarray,output_next = None, gradient = None):
        self._calculate_parital_gradient(y, output_next,gradient)
        if self.layer_type == Layer_type.Output:
            return self.partial_gradient*self.out_put
        if self.layer_type == Layer_type.Hidden:
            relu = Relu(self.input_data)
            return self.partial_gradient * relu.der() * self.input_data

    def _calculate_parital_gradient(self, label: np.ndarray, output_next = None,gradient= None):
        """
            does the backwards process to one object in the layer and returns the needed change (gradient)
            :param index gives the index for the object in the property cell 1 = neuron index cell 2 = weight index
            :param label the real value of the run cam also be the gradient of the last next layer
            :param output the output of the forward pass
            :param weights_next needed for the calculation of the gradient in the hidden layer
            :return float that has the change needed for the object
            :raise in case the index doesn't match the type
            """

        if self.layer_type == Layer_type.Output:
            if True in np.isnan(self.out_put):
                raise Exception('out put must not be null')
            softmx = softmax(self.out_put)
            softmx.regular()
            gradient = softmx.ds_dc(label, True)*softmx.ds_dw(True)
            self.partial_gradient = gradient
        elif self.layer_type == Layer_type.Hidden:
            if gradient is None or output_next is None:
                raise Exception('in hidden layer gradient and output_next must not be null')
            relu = Relu(output_next)
            self.partial_gradient = np.dot(np.squeeze(gradient).T, relu.der())
        else:
            raise Exception('Layer type must not be input layer')





