#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import namedtuple
import sys
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
class CNN(nn.Module):
    def __init__(self, e_char, e_word):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(CNN, self).__init__()
        self.e_word = e_word
        self.e_char = e_char
        self.convLayer = torch.nn.Conv1d(in_channels=self.e_char, out_channels=self.e_word, kernel_size=5, bias=True)
        self.maxpool = nn.MaxPool1d(17)

    def forward(self, x_reshaped):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """

        x_conv = self.convLayer(x_reshaped)
        return torch.squeeze(self.maxpool(nn.functional.relu(x_conv)), dim=2)

### END YOUR CODE

