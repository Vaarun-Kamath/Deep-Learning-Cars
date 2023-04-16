import numpy as np

class Layer:
    def __init__(self,n_inputs,n_neurons,activation):
        self.weights = 0.1*np.random.randn(n_neurons,n_inputs)
        print("===========")
        print(f"Initial Weights: (n_inputs,n_neurons) = ({n_inputs},{n_neurons})")
        print(self.weights)
        print("===========")
        self.biases =  np.zeros((n_neurons,1))
        self.activation = activation
    def forward_activate(self, inputs):
        # self.output = np.dot(self.weights,inputs) + self.biases
        self.reluflag = 0
        self.softflag = 0
        self.output = np.dot(self.weights,inputs)
        if self.activation == 'relu':
            self.output = np.maximum(0,self.output)
            # if self.reluflag < 2:
                # print("===========")
                # print(f"{self.reluflag} self.output: (relu) {self.output.shape}")
                # print(self.output)
                # print("===========")
                # self.reluflag += 1
        elif self.activation == 'softmax':
            exp_val = np.exp(self.output - np.max(self.output,axis=0,keepdims=False))
            self.output = exp_val / np.sum(exp_val,axis=0,keepdims=False)
            # if self.softflag < 1:
            #     # print("===========")
            #     # print(f"{self.softflag} self.output: (Softmax) {self.output.shape}")
            #     # print(self.output)
            #     # print("===========")
            #     self.softflag += 1

class NN:
    def __init__(self) -> None:
        self.L1 = Layer(5,6,'relu')
        self.L2 = Layer(6,6,'relu')
        self.L3 = Layer(6,4,'softmax')