import torch
from torch import nn

# Define the custom neural network
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        self.flatten = nn.Flatten()
        # Define layers of the neural network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Add more layers...
        self.fc1 = nn.Linear(256, 200) # 200 is the number of classes in TinyImageNet


    def forward(self, x):
        # Define forward pass
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = torch.relu(x)

        # x = self.global_pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x