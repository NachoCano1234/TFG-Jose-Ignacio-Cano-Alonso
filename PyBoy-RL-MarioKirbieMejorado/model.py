import torch
import torch.nn as nn
import copy


class DDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        c, h, w = input_dim

        self.feature = nn.Sequential(
            nn.Conv2d(c, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            n_flatten = self.feature(dummy).shape[1]

        self.value_stream = nn.Sequential(
            nn.Linear(n_flatten, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.adv_stream = nn.Sequential(
            nn.Linear(n_flatten, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

        self.online = nn.ModuleDict({
            "feature": self.feature,
            "value": self.value_stream,
            "adv": self.adv_stream
        })

        self.target = copy.deepcopy(self.online)

        for p in self.target.parameters():
            p.requires_grad = False


    def forward(self, x, model="online"):

        net = self.online if model == "online" else self.target

        features = net["feature"](x)

        value = net["value"](features)
        adv = net["adv"](features)

        q = value + (adv - adv.mean(dim=1, keepdim=True))

        return q