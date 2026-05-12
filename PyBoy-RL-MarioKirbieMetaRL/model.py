#Cambio por Nacho MetaRL. Se cambia todo el modelo
import torch
import torch.nn as nn
import copy

class ContextEncoder(nn.Module): #Cambio por NachoRL. Proporciona el contexto.
    def __init__(self, input_dim, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

    def forward(self, x):
        return self.net(x)


class MetaDDQN(nn.Module):
    def __init__(self, input_dim, action_dim, latent_dim=64):
        super().__init__()

        c, h, w = input_dim

        # CNN
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

        self.context_encoder = ContextEncoder(input_dim=n_flatten + action_dim + 1)

        # Q network condicionado
        self.q_net = nn.Sequential(
            nn.Linear(n_flatten + latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

        self.target = copy.deepcopy(self.q_net)

        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, state, context_z, model="online"):
        features = self.feature(state)

        x = torch.cat([features, 0.1 * context_z], dim=1) #Evita que z destruya el aprendizaje

        if model == "online":
            return self.q_net(x)
        else:
            return self.target(x)