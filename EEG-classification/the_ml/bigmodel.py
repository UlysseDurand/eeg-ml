# Model

from torch import nn

def model(x_example, hyperparameters):
    channels = x_example.shape[0]
    w, h = x_example.shape[1:]

    return nn.Sequential(
        nn.BatchNorm2d(channels),
        ConvLayer(channels, channels, 0.5), # (B, channels, h, w) -> (B, channels, h-2, w-2)
        ConvLayer(channels, channels, 0.2), # (B, channels, h-2, w-2) -> (B, channels, h-4, w-4)
        ConvLayer(channels, channels, 0.2), # (B, channels, h-4, w-4) -> (B, channels, h-6, w-6)
        nn.Flatten(),
        LinearLayer(channels*(h-6)*(w-6), 128, 0.5),
        LinearLayer(128, 4, 0.6)
    )

def ConvLayer(inputChannels, outputChannels, dropout):
    return nn.Sequential(
        nn.Conv2d(inputChannels, outputChannels, kernel_size=5, padding=1),
        nn.Dropout2d(dropout),
        nn.BatchNorm2d(outputChannels),
        nn.ReLU()
    )

def LinearLayer(input_layer_size, output_layer_size, dropout):
    return nn.Sequential(
        nn.Linear(input_layer_size, output_layer_size),
        nn.Dropout(0.6),
        nn.BatchNorm1d(output_layer_size),
        nn.ReLU()
    )