import torch
import torch.nn as nn
from math import sqrt

class MLP(nn.Module):
    """
    General multilayer perceptron to replicate the basic network that 
    was used in the physics-informed neural network paper
    """

    def __init__(self, n_hidden_layers=2,
                 n_neurons_per_layer=9,
                 activation=nn.Tanh, 
                 use_layernorm=False,
                 input=2, 
                 output=1
                 ) -> None:
        """
        Initialization for a multilayer perceptron using a sequence of nn.Linear, activation and (if true) layernorm
        layers to build the network 
        """

        super(MLP, self).__init__()

        #Build network
        layer_list = [nn.Linear(input, n_neurons_per_layer)] #(x,t) is default --> hidden layer size
        layer_list.append(activation())

        for i in range(n_hidden_layers):
            layer_list.append(nn.Linear(n_neurons_per_layer, n_neurons_per_layer))
            if (use_layernorm): layer_list.append(nn.LayerNorm(n_neurons_per_layer))
            layer_list.append(activation())
    
        layer_list.append(nn.Linear(n_neurons_per_layer, output)) #hidden layer size --> (u) is default

        self.model = nn.Sequential(*layer_list) #combine into a layer model
        self.initialize_weights()

    def initialize_weights(self) -> None:
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight) #Xavier normal initialization
                nn.init.zeros_(layer.bias) #Zero initialization for biases
            
    def forward(self, x, t) -> torch.Tensor:
        inputs = torch.cat([x, t], dim=1) #combine into a single layer tensor
        return self.model(inputs)
    

class SIREN(nn.Module):
    class SINE(nn.Module):
        """
        A module wrapper to use sin as activation function within a nn.Sequential block
        """
        def __init__(self, omega):
            super().__init__()
            self.omega = omega

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.sin(self.omega * x)
    
    def __init__(self, n_hidden_layers=2,
                 n_neurons_per_layer=9,
                 input=2, 
                 output=1,
                 nl_outer=False,
                 omega=30
                 ) -> None:
        """
        Initialization for a multilayer perceptron using a sequence of nn.Linear and sine activation functions, described as a SIREN network,
        this was implemented as a seperate class from the MLP due to it's use of a novel weight initialization scheme
        """

        super(SIREN, self).__init__()
        
        self.omega = omega #reference

        #Build network
        layer_list = [nn.Linear(input, n_neurons_per_layer)] #(x,t) is default --> hidden layer size
        layer_list.append(self.SINE(omega))

        for i in range(n_hidden_layers):
            layer_list.append(nn.Linear(n_neurons_per_layer, n_neurons_per_layer))
            layer_list.append(self.SINE(omega))
    
        layer_list.append(nn.Linear(n_neurons_per_layer, output)) #hidden layer size --> (u) is default
        if nl_outer: layer_list.append(self.SINE(omega)) #adds a nonlinear layer at the end, used in the paper, maybe needed

        self.initialize_weights(layer_list)
        self.model = nn.Sequential(*layer_list) #combine into a layer model
        self.initialize_weights()
    
    def initialize_weights(self):
        index = 0
        for layer in self.model:
            if isinstance(layer, nn.Linear):

                num_inputs = layer.weights.size(-1)

                if index == 0: 
                    layer.weight.uniform_(-1/num_inputs, 1,num_inputs)
                else: 
                    bound = sqrt(6 / num_inputs) / self.omega
                    layer.weight.uniform_(-bound, bound)

                index += 1
        

    def forward(self, x, t) -> torch.Tensor:
        inputs = torch.cat([x, t], dim=1) #combine into a single layer tensor
        return self.model(inputs)
        

    
