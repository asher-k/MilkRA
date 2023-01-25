import torch.nn as nn


class CNN(nn.Module):
    """
    Baseline CNN (as an intermediary between MLP & Transformer)
    """
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.mpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv2 = nn.Conv2d(in_channels=1, out_channels=50, kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.mpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.dense1 = nn.Linear(in_features=100, out_features=500)
        self.relu3 = nn.ReLU()

        self.dense2 = nn.Linear(in_features=500, out_features=num_classes)
        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self, x):
        x = nn.Linear(len(x))
        return x
