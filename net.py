import torch
from torch import nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, device=torch.device("cpu")):
        super(Net, self).__init__()
        self.device = device
        self.conv1 = torch.nn.Conv1d(12, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        self.fc1 = nn.Linear(256 + 128 + 64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(64)

    def forward(self, inp):
        x_inp1 = self.bn1(F.relu(self.conv1(inp)))      # (B, 64, n_points)
        x_inp2 = self.bn2(F.relu(self.conv2(x_inp1)))   # (B, 128, n_points)
        x_inp3 = self.bn3(F.relu(self.conv3(x_inp2)))   # (B, 256, n_points)

        x_inp1 = torch.mean(x_inp1, 2, keepdim=True)    # (B, 64, 1)
        feat_dim1 = x_inp1.size()[1]
        x_inp1 = x_inp1.view(-1, feat_dim1)             # (B, 64)

        x_inp2 = torch.mean(x_inp2, 2, keepdim=True)    # (B, 128, 1)
        feat_dim2 = x_inp2.size()[1]
        x_inp2 = x_inp2.view(-1, feat_dim2)             # (B, 128)

        x_inp3 = torch.mean(x_inp3, 2, keepdim=True)    # (B, 256, 1)
        feat_dim3 = x_inp3.size()[1]
        x_inp3 = x_inp3.view(-1, feat_dim3)             # (B, 256)

        x_inpt = torch.cat([x_inp1, x_inp2, x_inp3], -1)  # (B, 64 + 128 + 256)

        x_inp = self.bn4(F.relu(self.fc1(x_inpt)))      # (B, 128)
        x_inp = self.bn5(F.relu(self.fc2(x_inp)))       # (B, 64)
        x_inp = self.sigmoid(self.fc3(x_inp))
        return x_inp