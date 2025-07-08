import torch.nn as nn
import torch.nn.functional as F

class TumorClassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.con1d = nn.Conv2d(1, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.con2d = nn.Conv2d(32, 64, kernel_size=3)
        self.con3d = nn.Conv2d(64, 128, kernel_size=3)
        self.fc1 = nn.Linear(128 * 26 * 26, 512)
        self.fc2 = nn.Linear(512, 256)
        self.output = nn.Linear(256, 4)

    def forward(self, X):
        X = self.pool(F.relu(self.con1d(X)))
        X = self.pool(F.relu(self.con2d(X)))
        X = self.pool(F.relu(self.con3d(X)))
        X = X.view(-1, 128 * 26 * 26)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.output(X)
        return X

class GliomaStageModel(nn.Module):
    def __init__(self, n_features=9, hidden1=100, hidden2=50, hidden3=30, output=2):
        super().__init__()
        self.fc1 = nn.Linear(n_features, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.out = nn.Linear(hidden3, output)

    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = self.out(X)
        return X
