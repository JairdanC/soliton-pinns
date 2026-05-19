import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    General multilayer perceptron to replicate the basic network that 
    was used in the physics-informed neural network paper
    """

    def __init__(self, n_hidden_layers=2, n_neurons_per_layer=9, activation=nn.Tanh, use_layernorm=False, input=2, output=1):
        super(MLP, self).__init__()

        n_output_nodes = output #u is default
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

    def initialize_weights(self):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight) #Xavier normal initialization
                nn.init.zeros_(layer.bias) #Zero initialization for biases
            
    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1) #combine into a single layer tensor
        return self.model(inputs)