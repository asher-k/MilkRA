import torch.nn as nn
from torch import flatten, cat


class CMapNN(nn.Module):
    """
    Fully-convolutional NN with final dense layers replaced by Global Average Pooling; this enables the computation
    of Class Activation Maps (CAMs) and reduces the number of free parameters in the model.
    """
    def __init__(self, num_classes, kernel_size, pad_mode='circular'):
        super(CMapNN, self).__init__()
        # Conv block 1; output 16c
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=kernel_size, stride=1, padding_mode=pad_mode,
                               padding=1)
        self.relu1 = nn.ReLU()
        self.norm1 = nn.BatchNorm2d(self.conv1.out_channels)
        self.mpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv block 2; output 32c
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=kernel_size, stride=1, padding_mode=pad_mode,
                               padding=1)
        self.relu2 = nn.ReLU()
        self.norm2 = nn.BatchNorm2d(self.conv2.out_channels)
        self.mpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv block 3; output 64c
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel_size, stride=1, padding_mode=pad_mode,
                               padding=1)
        self.relu3 = nn.ReLU()
        self.norm3 = nn.BatchNorm2d(self.conv3.out_channels)
        self.mpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # GAP, Dense & SM
        self.d = nn.Linear(in_features=64, out_features=num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, early_stopping=False):
        """
        Performs a forward pass of the model.

        :param x: Tensor data sample
        :param early_stopping: Enables early stopping of the model after the final convolutional layer
        :return: Int predicted class, unused auxiliary parameter
        """
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
    Baseline CNN leveraging LeNet architecture.
    """
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        # Conv block 1; output 16c
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 5), stride=(2, 2),
                               padding_mode='circular', padding=3)
        self.relu1 = nn.ReLU()
        self.norm1 = nn.BatchNorm2d(self.conv1.out_channels)
        self.mpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv block 2; output ???
        self.conv2 = nn.Conv2d(in_channels=self.conv1.out_channels//self.mpool1.kernel_size, out_channels=32,
                               kernel_size=(5, 5), stride=(2, 2), padding_mode='circular', padding=3)
        self.relu2 = nn.ReLU()
        self.norm2 = nn.BatchNorm2d(self.conv2.out_channels)
        self.mpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dense layer 1
        self.dense1 = nn.Linear(in_features=272*16, out_features=64)
        self.relu3 = nn.ReLU()
        self.norm3 = nn.BatchNorm1d(self.dense1.out_features)

        # Dense layer 2
        self.dense2 = nn.Linear(in_features=self.dense1.out_features, out_features=num_classes)
        self.norm4 = nn.BatchNorm1d(self.dense2.out_features)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        """
        Performs a forward pass of the model.

        :param x: Tensor data sample
        :return: Int predicted class, unused auxiliary parameter
        """
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
