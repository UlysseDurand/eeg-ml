# Model

from torch import nn

def model(x_example, hyperparameters):
    num_channels = x_example.shape[0]
    w, h = x_example.shape[1:]

    return nn.Sequential(
        nn.BatchNorm2d(num_channels),
        ConvLayer(num_channels, num_channels*4, 0.3, 5), # (B, num_channels, h, w) -> (B, num_channels, h-2, w-2)
        ConvLayer(num_channels*4, num_channels*2, 0.2, 5), # (B, num_channels, h-2, w-2) -> (B, num_channels, h-4, w-4)
        ConvLayer(num_channels*2, num_channels, 0.2, 5), # (B, num_channels, h-4, w-4) -> (B, num_channels, h-6, w-6)
        nn.Flatten(),
        LinearLayer(num_channels*h*w, 128, 0.5),
        LinearLayer(128, 4, 0.6)
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
