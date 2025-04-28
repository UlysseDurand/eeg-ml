# Model

from torch import nn

model = nn.Sequential(
    nn.BatchNorm2d(32),
    nn.Conv2d(32, 32, kernel_size=5, padding=1),  # (B,32,5,101) → (B,32,3,99)
    nn.Dropout2d(0.5),
    nn.BatchNorm2d(32),
    nn.ReLU(),

    nn.Conv2d(32, 32, kernel_size=3, padding=0),  # (B,32,3,99) → (B,32,1,97)
    nn.Dropout2d(0.5),
    nn.BatchNorm2d(32),
    nn.ReLU(),

    nn.Conv2d(32, 8, kernel_size=11, padding=5),
    nn.Dropout(0.5),
    nn.BatchNorm2d(8),

    nn.Flatten(),

    nn.Linear(8*97, 128),
    nn.Dropout(0.5),
    nn.BatchNorm1d(128),
    nn.ReLU(),

    nn.Linear(128, 4)
)