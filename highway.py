#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import namedtuple
import sys
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
import numpy as np

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
class Highway(nn.Module):
    def __init__(self, embed_size):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Highway, self).__init__()
        self.embed_size = embed_size
        self.gate_layer = torch.nn.Linear(in_features=self.embed_size, out_features=self.embed_size, bias=True)
        self.proj_layer = torch.nn.Linear(in_features=self.embed_size, out_features=self.embed_size, bias=True)

    def forward(self, x_conv_out):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        x_proj = F.relu(self.proj_layer(x_conv_out))
        x_gate = torch.sigmoid(self.gate_layer(x_conv_out))
        return torch.mul(x_proj,x_gate) + torch.mul((1 - x_gate),x_conv_out)

### END YOUR CODE 

