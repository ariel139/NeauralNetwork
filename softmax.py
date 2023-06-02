import numpy as np


class softmax:
    def __init__(self, input_vector: np.ndarray, index= None):
        self.index = index
        self.input_vector = input_vector
        self.output_vector = None

    @staticmethod
    def kr_delta(i, j):
        return 1 if i == j else 0

    def regular(self,temp = 1.0 ):
        stable_input = self.input_vector / temp
        upper = np.exp(stable_input)
        self.output_vector = upper / np.sum(upper)
        return self.output_vector[self.index] if self.index is not None else self.output_vector

    def ds_dc(self, label: np.ndarray, return_index = False):
        res = self.output_vector - label
        return res[self.index] if not return_index else res

    def ds_dw(self,  return_index=False):
        if self.output_vector is None:
            raise Exception('must run the regular method first')
        # l = [self.output_vector[i]*(self.kr_delta(i,k_vector[i])- self.regular(i) for i in k_vector]
        res = np.array([self.output_vector[self.index]*(self.kr_delta(i,self.index)-self.output_vector[i]) for i in range(len(self.output_vector))])
        return res[self.index] if not return_index else res





