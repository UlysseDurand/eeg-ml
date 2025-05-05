# Model

from torch import nn

def model(x_example, hyperparameters):
    num_channels = x_example.shape[0]
    w, h = x_example.shape[1:]

    return nn.Sequential(
        nn.BatchNorm2d(num_channels),
        ConvLayer(num_channels, num_channels*4, 0.5, 5),
        ConvLayer(num_channels*4, num_channels*2, 0.4, 5),
        ConvLayer(num_channels*2, num_channels, 0.4, 5),
        nn.Flatten(),
        LinearLayer(num_channels*h*w, 128, 0.75),
        nn.Linear(128, 4)
    )

def ConvLayer(inputChannels, outputChannels, dropout, ks):
    return nn.Sequential(
        nn.Conv2d(inputChannels, outputChannels, kernel_size=ks, padding=int((ks-1)/2)),
        nn.Dropout2d(dropout),
        nn.BatchNorm2d(outputChannels),
        nn.ReLU()
    )

def LinearLayer(input_layer_size, output_layer_size, dropout):
    return nn.Sequential(
        nn.Linear(input_layer_size, output_layer_size),
        nn.Dropout(dropout),
        nn.BatchNorm1d(output_layer_size),
        nn.ReLU()
    )
