# edge/model.py
import torch
import torch.nn as nn

class MultiExitNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Shared layers
        self.conv1 = nn.Conv1d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1)

        # Exit heads
        self.exit1 = nn.Linear(16, num_classes)   # Shallow exit
        self.exit2 = nn.Linear(32, num_classes)   # Mid exit
        self.exit3 = nn.Linear(64, num_classes)   # Deep exit

    def forward(self, x, exit_depth=3):
        x = torch.relu(self.conv1(x))
        if exit_depth == 1:
            return self.exit1(x.mean(dim=2))  

        x = torch.relu(self.conv2(x))
        if exit_depth == 2:
            return self.exit2(x.mean(dim=2))  

        x = torch.relu(self.conv3(x))
        return self.exit3(x.mean(dim=2))
