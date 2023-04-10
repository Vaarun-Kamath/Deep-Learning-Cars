import numpy as np


# nnfs.init()

# np.random.seed(0)

# #Inputs
# X = [[1, 2, 3],
#      [2.0, 5.0, -1.0]]

# # X,y = spiral_data(100,3)

# inputs = [0,2,-1,3.3,2.7,1.1,2.2,-100]
# output = []

# class Layer:
#     def __init__(self,n_inputs,n_neurons):
#         self.weights = 0.1*np.random.randn(n_inputs, n_neurons)
#         self.biases =  np.zeros((1,n_neurons) )
#     def forward(self, inputs):
#         self.output = np.dot(inputs,self.weights) + self.biases

# class activation_relu:
#     def forward(self,inputs):
#         self.output = np.maximum(0,inputs)

# class activation_softmax:
#     def forward(self,inputs):
#         exp_val = np.exp(inputs - np.max(inputs,axis=1,keepdims=True)) 
#         probab = exp_val / np.sum(exp_val,axis=1,keepdims=True)
#         self.output = probab

# class Loss:
#     def calculate(self,output, y):
#         sample_losses = self.forward(output,y)
#         data_loss = np.mean(sample_losses)
#         return data_loss

# class Loss_CategoricalCrossentropy(Loss):
#     def forward(self,y_pred,y_true):
#         samples = len(y_pred)
#         y_pred_clipped = np.clip(y_pred,1e-7,1-1e-7)
#         if len(y_true.shape) == 1:
#             correct_confidences = y_pred_clipped[range(samples),y_true]
#         elif len(y_true.shape) == 2:
#             correct_confidences = np.sum(y_pred_clipped*y_true,axis =1)
#         negative_log_likelihoods = -np.log(correct_confidences)
#         return negative_log_likelihoods

# X,y = spiral_data(100,3)

# dense1 = Layer(2,3)
# activation1 = activation_relu()

# dense2 = Layer(3,3)
# activation2 = activation_softmax()

# dense1.forward(X)
# activation1.forward(dense1.output)

# dense2.forward(activation1.output)
# activation2.forward(dense2.output)

# print(activation2.output[:5])

# loss_function = Loss_CategoricalCrossentropy()
# loss = loss_function.calculate(activation2.output, y)
# print("LOSS: ",loss)


class Layer:
    def __init__(self,n_inputs,n_neurons,activation):
        self.weights = 0.1*np.random.randn(n_neurons,n_inputs)
        print("===========")
        print(f"(n_inputs,n_neurons) = ({n_inputs},{n_neurons})")
        print(self.weights)
        print("===========")
        self.biases =  np.zeros((n_neurons,1))
        self.activation = activation
    def forward_activate(self, inputs):
        # self.output = np.dot(self.weights,inputs) + self.biases
        self.output = np.dot(self.weights,inputs)
        if self.activation == 'relu':
            self.output = np.maximum(0,self.output)
        elif self.activation == 'softmax':
            exp_val = np.exp(self.output - np.max(self.output,axis=0,keepdims=False))
            self.output = exp_val / np.sum(exp_val,axis=0,keepdims=False)

class NN:
    def __init__(self) -> None:
        self.L1 = Layer(5,6,'relu')
        self.L2 = Layer(6,6,'relu')
        self.L3 = Layer(6,4,'softmax')
