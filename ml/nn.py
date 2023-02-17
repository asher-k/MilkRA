import torch
import torch.nn as nn
from torch import flatten, cat


class WindowCNN(nn.Module):
    """
    CNN loosely based on LeNet architecture that subdivides droplet images into 31x31 sections for individual processing
    """
    def __init__(self, num_classes):
        super(WindowCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(5, 5), padding_mode='circular', padding=3)
        self.relu1 = nn.ReLU()
        self.mpool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))

        self.conv2 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(5, 5), padding_mode='circular', padding=3)
        self.relu2 = nn.ReLU()
        self.mpool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))

        self.conv3 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(5, 5), padding_mode='circular', padding=3)
        self.relu3 = nn.ReLU()
        self.mpool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))

        self.conv4 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(5, 5), padding_mode='circular', padding=3)
        self.relu4 = nn.ReLU()
        self.mpool4 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))

        self.dense1 = nn.Linear(in_features=968, out_features=64)
        self.relu3 = nn.ReLU()
        self.dense2 = nn.Linear(in_features=64, out_features=num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x1, x2, x3, x4 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]

        c1 = self.conv1(x1)
        c1 = self.relu1(c1)
        c1 = self.mpool1(c1)
        c1f = flatten(c1, 1)

        c2 = self.conv2(x2)
        c2 = self.relu2(c2)
        c2 = self.mpool2(c2)
        c2f = flatten(c2, 1)

        c3 = self.conv3(x3)
        c3 = self.relu3(c3)
        c3 = self.mpool3(c3)
        c3f = flatten(c3, 1)

        c4 = self.conv4(x4)
        c4 = self.relu4(c4)
        c4 = self.mpool4(c4)
        c4f = flatten(c4, 1)

        x = cat([c1f, c2f, c3f, c4f], dim=1)
        x = self.dense1(x)
        x = self.relu3(x)
        x = self.dense2(x)
        predictions = self.softmax(x)
        return predictions, (c1, c2, c3, c4)  # returns predictions, convolved & pooled images


class CMapNN(nn.Module):
    def __init__(self, num_classes, ks):
        """
        Fully-convolutional NN with final dense layers replaced by Global Average Pooling
        """
        super(CMapNN, self).__init__()
        # Conv block 1; output 16c
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=ks, stride=1, padding_mode='circular',
                               padding=1)
        self.relu1 = nn.ReLU()
        self.norm1 = nn.BatchNorm2d(self.conv1.out_channels)
        self.mpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv block 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=ks, stride=1, padding_mode='circular',
                               padding=1)
        self.relu2 = nn.ReLU()
        self.norm2 = nn.BatchNorm2d(self.conv2.out_channels)
        self.mpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv block 3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=ks, stride=1, padding_mode='circular',
                               padding=1)
        self.relu3 = nn.ReLU()
        self.norm3 = nn.BatchNorm2d(self.conv3.out_channels)
        self.mpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # GAP, Dense & SM
        self.d = nn.Linear(in_features=64, out_features=num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, early_stopping=False):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.norm1(x)
        x = self.mpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.norm2(x)
        x = self.mpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.norm3(x)
        x = self.mpool3(x)

        if early_stopping:  # For direct extraction of convolved sample
            return x, None

        x = x.view(x.shape[0], 64, x.shape[2]*x.shape[3]).mean(2)  # GAP reshape to [BS, 64, IMG]
        x = self.d(x)
        predictions = self.softmax(x)
        return predictions, None


class CNN(nn.Module):
    """
    Baseline CNN (as an intermediary between MLP & Transformer) using LeNet architecture
    """
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 5), stride=(2, 2),
                               padding_mode='circular', padding=3)
        self.relu1 = nn.ReLU()
        self.norm1 = nn.BatchNorm2d(self.conv1.out_channels)
        self.mpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=self.conv1.out_channels//self.mpool1.kernel_size, out_channels=32,
                               kernel_size=(5, 5), stride=(2, 2), padding_mode='circular', padding=3)
        self.relu2 = nn.ReLU()
        self.norm2 = nn.BatchNorm2d(self.conv2.out_channels)
        self.mpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dense1 = nn.Linear(in_features=272*16, out_features=64)
        self.relu3 = nn.ReLU()
        self.norm3 = nn.BatchNorm1d(self.dense1.out_features)

        self.dense2 = nn.Linear(in_features=self.dense1.out_features, out_features=num_classes)
        self.norm4 = nn.BatchNorm1d(self.dense2.out_features)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.norm1(x)
        x = self.mpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.norm2(x)
        x = self.mpool2(x)

        x = flatten(x, 1)
        x = self.dense1(x)
        x = self.relu3(x)
        x = self.norm3(x)

        x = self.dense2(x)
        x = self.norm4(x)
        predictions = self.softmax(x)
        return predictions, None  # does not return convolutional filters at a timestep
