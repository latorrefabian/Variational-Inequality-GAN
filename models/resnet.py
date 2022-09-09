#  MIT License

# Copyright (c) Facebook, Inc. and its affiliates.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# written by Hugo Berard (berard.hugo@gmail.com) while at Facebook.

import torch
import torch.nn as nn
from .discriminator import Discriminator
from typing import Optional


class ResBlock(nn.Module):
    def __init__(self, num_filters, resample=None, batchnorm=True, inplace=False):
        super(ResBlock, self).__init__()

        if resample == 'up':
            conv_list = [nn.ConvTranspose2d(num_filters, num_filters, 4, stride=2, padding=1),
                        nn.Conv2d(num_filters, num_filters, 3, padding=1)]
            self.conv_shortcut =  nn.ConvTranspose2d(num_filters, num_filters, 1, stride=2, output_padding=1)

        elif resample == 'down':
            conv_list = [nn.Conv2d(num_filters, num_filters, 3, padding=1),
                        nn.Conv2d(num_filters, num_filters, 3, stride=2, padding=1)]
            self.conv_shortcut = nn.Conv2d(num_filters, num_filters, 1, stride=2)

        elif resample == None:
            conv_list = [nn.Conv2d(num_filters, num_filters, 3, padding=1),
                        nn.Conv2d(num_filters, num_filters, 3, padding=1)]
            self.conv_shortcut = None
        else:
            raise ValueError('Invalid resample value.')

        self.block = []
        for conv in conv_list:
            if batchnorm:
                self.block.append(nn.BatchNorm2d(num_filters))
            self.block.append(nn.ReLU(inplace))
            self.block.append(conv)

        self.block = nn.Sequential(*self.block)

    def forward(self, x):
        shortcut = x
        if not self.conv_shortcut is None:
            shortcut = self.conv_shortcut(x)
        return shortcut + self.block(x)


class ResNet32Generator(nn.Module):
    def __init__(
            self,
            n_in: int,
            n_out: int,
            num_filters: int = 128,
            batchnorm: bool = True,
            number_of_classes: Optional[int] = None):
        super(ResNet32Generator, self).__init__()
        self.num_filters = num_filters
        self.input = nn.Linear(n_in, 4*4*num_filters)
        self.network = [
            ResBlock(num_filters, resample='up', batchnorm=batchnorm, inplace=True),
            ResBlock(num_filters, resample='up', batchnorm=batchnorm, inplace=True),
            ResBlock(num_filters, resample='up', batchnorm=batchnorm, inplace=True)]
        if batchnorm:
            self.network.append(nn.BatchNorm2d(num_filters))
        self.network += [
            nn.ReLU(True),
            nn.Conv2d(num_filters, 3, 3, padding=1),
            nn.Tanh()]

        self.network = nn.Sequential(*self.network)
        if number_of_classes is not None:
            self.label_embedding = LabelEmbedding(
                number_of_classes=number_of_classes,
                embedding_dimension=[n_in])
        else:
            self.label_embedding = lambda x, _: x

    def forward(self, noise, labels=None):
        if labels is not None:
            noise = self.label_embedding(noise, labels)
        x = self.input(noise).view(len(noise), self.num_filters, 4, 4)
        return self.network(x)


class ResNet32Discriminator(nn.Module):
    def __init__(
            self,
            n_in: int,
            n_out: int,
            num_filters: int = 128,
            batchnorm: bool = False,
            number_of_classes: Optional[int] = None):
        super(ResNet32Discriminator, self).__init__()
        self.block1 = nn.Sequential(
            nn.LazyConv2d(num_filters, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_filters, num_filters, 3, stride=2, padding=1))
        self.shortcut1 = nn.LazyConv2d(num_filters, 1, stride=2)
        self.network = nn.Sequential(
            ResBlock(num_filters, resample='down', batchnorm=batchnorm),
            ResBlock(num_filters, resample=None, batchnorm=batchnorm),
            ResBlock(num_filters, resample=None, batchnorm=batchnorm),
            nn.ReLU())
        self.output = nn.Linear(num_filters, 1)
        if number_of_classes is not None:
            self.label_embedding = LabelEmbedding(
                number_of_classes=number_of_classes,
                embedding_dimension=[1, 32, 32])
        else:
            self.label_embedding = lambda x, _: x

    def forward(self, x, labels=None):
        if labels is not None:
            x = self.label_embedding(x, labels)
        y = self.block1(x)
        y = self.shortcut1(x) + y
        y = self.network(y).mean(-1).mean(-1)
        y = self.output(y)

        return y


class LabelEmbedding(nn.Module):
    """
    Embedding for an integer label in a supervised dataset

    Attributes:
        embedding: tensor containing the embeddings, one for each label.
        module: module that will receive the data and the embedding of
          the label as input.
    """
    def __init__(
            self,
            number_of_classes: int,
            embedding_dimension: list[int]) -> None:
        """Initializes the Label Embedding"""
        super().__init__()
        parameter_dimension = [
            number_of_classes, *embedding_dimension]
        self.embedding = nn.Parameter(
            torch.randn(*parameter_dimension))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Stacks the data and the embedding of the label"""
        return torch.hstack((x, self.embedding[y]))
