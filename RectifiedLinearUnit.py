import numpy as np


class Relu:
	def __init__(self,input_vector:np.ndarray, index = None):
		self.input_vector = input_vector
		self.index = index
		self.output_vector = None

	def regular(self):
		self.output_vector = np.maximum(self.input_vector,0)

	def der(self):
		return np.greater(self.input_vector, 0.).astype(np.float32)
