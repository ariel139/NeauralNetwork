import numpy as np
import random
from Layer_type import Layer_type


class Neural_Network:
    def __init__(self,batch_size, learning_rate: float, data: np.ndarray, true_data = np.ndarray,sgd= True) -> None:
        """
        initiates the properties of the neural network class
        :param learning_rate the learning rate for the model
        :param data  the input data for the model
        :raises TypeError when gets none valid input and when dims don't have at least two layers
        """
        if not isinstance(learning_rate, float):
            raise TypeError(f'None Valid Data Type!\ngot {type(learning_rate)} instead of a float')

        if not isinstance(data, np.ndarray):
            raise TypeError(f'None Valid Data Type!\ngot {type(data)} instead of a numpy.ndarray')

        self.learning_rate = learning_rate
        self.data = data
        self.layers = []
        self.sgd = sgd
        self.batch_size = batch_size
        self.true_data = true_data

    def set_layers(self, layers: list):
        self.layers = layers
        if layers[0].input_data is not None and self.sgd:
            print('WARNING: in SGD mode the input data in splited to batches by the program.')

    def _forward(self,sample: np.ndarray):
        if self.layers == []:
            raise Exception('self.layers must not be None')
        self.layers[0].input_data = sample.reshape((self.layers[0].input_size,))
        output = None
        for layer in self.layers:

            if layer.layer_type != Layer_type.Input or layer.input_data is None:
                layer.input_data = output
                layer.forward()
            layer.forward()
            output = layer.out_put

        return self.layers[-1].out_put

    def _backprop(self,true_sample: np.ndarray):
        gradient = []
        prg = None
        for layer_index in reversed(range(len(self.layers))):
            if self.layers[layer_index].layer_type != Layer_type.Input:
                if self.layers[layer_index].layer_type== Layer_type.Hidden:
                    loss = self.layers[layer_index].backward(true_sample, self.layers[layer_index+1].input_data, prg)
                else:
                    loss = self.layers[layer_index].backward(true_sample)
                prg = self.layers[layer_index].partial_gradient
                gradient.append(loss)
        return gradient

    def _split_data(self):
        self.data = np.array_split(self.data,self.batch_size)
        self.true_data = np.array_split(self.true_data,self.batch_size)

    def _change_weights(self, gradient):
        change = self.learning_rate*gradient
        for i in range(len(self.layers)):
            self.layers[i].weights -= change[i]

    @staticmethod
    def evaluate_accuracy(y, y_true):
        return np.sum(np.power(y_true - y, 2))

    def train(self, epoches: int):
        if self.sgd:
            self._split_data()
        for i in range(epoches):
            train_data = list(zip(self.data, self.true_data))
            random.shuffle(train_data)
            for batch, true_batch in train_data:
                gradient_vec = None
                for sample,true_sample in zip(batch, true_batch):
                    self._forward(sample)
                    gradient = self._backprop(true_sample)
                    if gradient_vec is None:
                        gradient_vec = gradient
                    else:
                        gradient_vec+gradient
                gradient_vec = gradient_vec / self.batch_size
                self._change_weights(gradient_vec)
            y = train_data[0][0]
            y_true = train_data[0][0]
            print(f'EPOCH number {i}/{epoches} accuracy = {self.evaluate_accuracy(y,y_true)}')



