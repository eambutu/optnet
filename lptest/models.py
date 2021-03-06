import torch

import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from qpth.qp import QPFunction

class OptNetEq(nn.Module):
    def __init__(self, nVar, nCon, Qpenalty, cuda=True):
        super().__init__()
        if not cuda:
            self.Q = Variable(Qpenalty * torch.eye(nVar).double())
            self.G = Variable(-torch.eye(nVar).double())
            self.h = Variable(torch.zeros(nVar).double())
            self.A = Parameter(torch.rand(nCon, nVar).double())
            self.b = Variable(torch.ones(nCon).double())
        else:
            self.Q = Variable(Qpenalty * torch.eye(nVar).double().cuda())
            self.G = Variable(-torch.eye(nVar).double().cuda())
            self.h = Variable(torch.zeros(nVar).double().cuda())
            self.A = Parameter(torch.rand(nCon, nVar).double().cuda())
            self.b = Variable(torch.ones(nCon).double().cuda())


    def forward(self, x):
        return QPFunction(verbose=-1)(
            self.Q, x.double(), self.G, self.h, self.A, self.b
        ).float()
