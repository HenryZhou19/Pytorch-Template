from torch import nn


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 100),
            nn.BatchNorm1d(100),
            nn.SiLU(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.SiLU(),
        )

    def forward(self, x):
        x = self.net(x)
                
        return x