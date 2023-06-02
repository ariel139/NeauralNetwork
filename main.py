import numpy as np
from NeuralNetwork import Neural_Network
import matplotlib.pyplot as plt
from Layer import Layer, Layer_type




#
#
# m =50
# colors = np.random.randint(0, 256, (m, 3)) / 255.0  # Random RGB colors for each point, divided by 255
# # Create the figure and axis
# fig, ax = plt.subplots()
#
# # Plot the scatter plot with RGB colors
#
# # Set labels and title
# ax.set_xlabel('x')
# ax.set_ylabel('loss')
# ax.set_title('loss with respect to weight')
#
# # Display the plot
#
# layer = Layer()
# y_true = np.transpose(np.eye(3,1))[0]
# lr = 0.1
# weights = []
# losss = []
# last =  np.random.random(5)
# for i in range(m):
# 	pred = layer.forward()
# 	change = layer._backward_object(object_type.weight, (0, 0), pred, y_true,last)
# 	cost = np.sum(np.power(y_true-pred,2))
# 	weight = layer.weights- lr*change[0]
# 	layer.weights[0][0] = weight[0][0]
# 	weights.append(weight[0][0])
# 	losss.append(cost)
#
# ax.plot(weights, losss, label='weight', color= 'red')
# ax.plot(range(m), losss, label= 'iteration')
# # plt.xlim(0,5)
# # plt.ylim(0,100)
# # ax.plot(weights, losss)
#
# plt.show()

data = np.load('./testing/mnist.npz')

nur = Neural_Network(100,0.1,data['x_train'],data['y_train'])
nur.set_layers([
	Layer(784,16,Layer_type.Input),
	Layer(16,16,Layer_type.Hidden),
	Layer(16,10,Layer_type.Hidden),
	Layer(10,16,Layer_type.Output)
])
nur.train(10)