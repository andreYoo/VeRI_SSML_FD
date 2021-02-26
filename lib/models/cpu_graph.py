import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
import numpy as np
device_0 = torch.device('cuda:0')
device_1 = torch.device('cuda:1')
device_cpu = torch.device('cpu')
import pdb

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
            self.memory[y] = self.memory[y] + x
            self.memory[y] /= self.memory[y].norm()

        #self.memory = self.global_norm(self.memory.data)
        return grad_inputs, None

class cpu_Graph(nn.Module):
    def __init__(self, num_features, num_classes, use_dram=False, alpha=0.01):
        super(cpu_Graph, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.alpha = alpha
        self.use_dram = use_dram
        self.global_norm = nn.BatchNorm1d(num_features)
        if self.use_dram==True:
            self.mem = nn.Parameter(torch.zeros(num_classes, num_features), requires_grad=False).to(device_cpu)
        else:
            self.mem = nn.Parameter(torch.zeros(num_classes, num_features), requires_grad=False)
        self.tpid_memory = np.zeros([num_classes],dtype=np.uint32)

    def store(self,inputs,target,device=device_1):
        self.mem[target]  = inputs.to(device)

    def global_normalisation(self):
        self.mem.data = self.global_norm(self.mem.data)
        self.mem.data /= self.mem.data.norm()

    def forward(self, inputs, targets,device=device_1, epoch=None):
        #originally, no device assignment
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = MemoryLayer(self.mem)(inputs, targets)
        return logits.to(device_1)
