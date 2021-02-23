import torch.nn as nn

""" Multi-Layer Neural Network"""


class DNet(nn.Module):
    def __init__(self, net_inputs, net_output, activation):
        super(DNet, self).__init__()
        self.fc1 = nn.Linear(net_inputs, net_output)
        self.act = activation

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        return x
