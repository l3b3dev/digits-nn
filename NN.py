import torch.nn as nn

""" Multi-Layer Neural Network for MLNN#1
    
    :arg net_inputs -- number of inputs, flattened out 16x16 image
    :arg num_classes - number of classes, 65 characters from given input
    
    One hidden layer with 1000 nodes, LogSoftmax activation for final classification
    
"""


class DClassNet(nn.Module):
    def __init__(self, net_inputs, num_classes):
        super(DClassNet, self).__init__()
        self.mlnn = nn.Sequential(
            nn.Linear(net_inputs, 1000),
            nn.ReLU(),
            nn.Linear(1000, num_classes),
            nn.LogSoftmax(dim=1))

    def forward(self, x):
        x = self.mlnn(x)
        return x


""" Multi-Layer Neural Network for MLNN#2

    :arg net_inputs -- number of inputs, flattened out 16x16 image

    One hidden layer with 1000 nodes, Sigmoid activation

"""


class DNet(nn.Module):
    def __init__(self, net_inputs):
        super(DNet, self).__init__()
        self.mlnn = nn.Sequential(
            nn.Linear(net_inputs, 1000),
            nn.ReLU(),
            nn.Linear(1000, net_inputs),
            nn.Sigmoid())

    def forward(self, x):
        x = self.mlnn(x)
        return x
