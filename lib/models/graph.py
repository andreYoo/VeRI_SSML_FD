import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
import numpy as np
device_0 = torch.device('cuda:0')
device_1 = torch.device('cuda:1')

class MemoryLayer(Function):
    def __init__(self, memory):
        super(MemoryLayer, self).__init__()
        self.memory = memory
        self.global_norm = nn.BatchNorm1d(2048)

    def forward(self, inputs, targets):
        self.save_for_backward(inputs, targets)
        outputs = inputs.mm(self.memory.t())
        return outputs

    def backward(self, grad_outputs):
        inputs, targets = self.saved_tensors
        grad_inputs = None
        if self.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(self.memory)
        for x, y in zip(inputs, targets):
            self.memory[y] = 0.5*self.memory[y] + 0.5*x
            self.memory[y] /= self.memory[y].norm()

        #self.memory = self.global_norm(self.memory.data)
        return grad_inputs, None

class Graph(nn.Module):
    def __init__(self, num_features, num_classes, alpha=0.01):
        super(Graph, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.alpha = alpha
        self.global_norm = nn.BatchNorm1d(num_features)
        self.mem = nn.Parameter(torch.zeros(num_classes, num_features), requires_grad=False)
        self.tpid_memory = np.zeros([num_classes],dtype=np.uint32)

    def store(self,inputs,target):
        self.mem[target]  = inputs.to(device_1)

    def global_normalisation(self):
        self.mem.data = self.global_norm(self.mem.data)
        self.mem.data /= self.mem.data.norm()

    def forward(self, inputs, targets, epoch=None):
        #originally, no device assignment
        inputs = inputs.to(device_1)
        targets = targets.to(device_1)
        logits = MemoryLayer(self.mem)(inputs, targets)
        return logits
