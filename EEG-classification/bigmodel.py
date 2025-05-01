# Model

from torch import nn

def model(x_example, hyperparameters):
    channels = x_example.shape[0]
    w, h = x_example.shape[1:]

    return nn.Sequential(
        nn.BatchNorm2d(channels),
        ConvLayer(channels, channels), # (B, channels, h, w) -> (B, channels, h-2, w-2)
        ConvLayer(channels, channels), # (B, channels, h-2, w-2) -> (B, channels, h-4, w-4)
        ConvLayer(channels, channels), # (B, channels, h-4, w-4) -> (B, channels, h-6, w-6)
        ConvLayer(channels, channels), # (B, channels, h-6, w-6) -> (B, channels, h-8, w-8)
        ConvLayer(channels, channels), # (B, channels, h-8, w-8) -> (B, channels, h-10, w-10)
        ConvLayer(channels, channels), # (B, channels, h-10, w-10) -> (B, channels, h-12, w-12)
        ConvLayer(channels, channels), # (B, channels, h-12, w-12) -> (B, channels, h-14, w-14)
        ConvLayer(channels, channels), # (B, channels, h-14, w-14) -> (B, channels, h-16, w-16)
        nn.Flatten(),
        LinearLayer(channels*(h-16)*(w-16), 128),
        LinearLayer(128, 4)
    )

def ConvLayer(inputChannels, outputChannels):
    return nn.Sequential(
        nn.Conv2d(inputChannels, outputChannels, kernel_size=5, padding=1),
        nn.Dropout2d(0.2),
        nn.BatchNorm2d(outputChannels),
        nn.ReLU()
    )

def LinearLayer(input_layer_size, output_layer_size):
    return nn.Sequential(
        nn.Linear(input_layer_size, output_layer_size),
        nn.Dropout(0.5),
        nn.BatchNorm1d(output_layer_size),
        nn.ReLU()
    )