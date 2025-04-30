# Model

from torch import nn

def model(x_example):
    channels = x_example.shape[0]
    w, h = x_example.shape[1:]

    channels_2 = 8

    return nn.Sequential(
        nn.BatchNorm2d(channels),
        nn.Conv2d(channels, channels, kernel_size=5, padding=1),  # (B,channels,h,w) → (B,channels,h-2,w-2)
        nn.Dropout2d(0.2),
        nn.BatchNorm2d(channels),
        nn.ReLU(),

        nn.Conv2d(channels, channels, kernel_size=3, padding=0),  # (B,channels,h-2,w-2) → (B,channels,h-4,w-4)
        nn.Dropout2d(0.2),
        nn.BatchNorm2d(channels),
        nn.ReLU(),

        nn.Conv2d(channels, channels_2, kernel_size=11, padding=5), # (B, channels,h-4,w-4) → (B, channels_2,h-4,w-4) 
        nn.Dropout(0.2),
        nn.BatchNorm2d(channels_2),

        nn.Flatten(),

        nn.Linear(channels_2 * (w-4) * (h-4), 128),
        nn.Dropout(0.5),
        nn.BatchNorm1d(128),
        nn.ReLU(),

        nn.Linear(128, 4)
    )