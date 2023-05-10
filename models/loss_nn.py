import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_size=10, out_size=1, hidden_dim=32, norm_reduce=False):
        super(MLP, self).__init__()
        self.norm_reduce = norm_reduce
        self.model = nn.Sequential(
            nn.Linear(in_size, hidden_dim),
            # nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_size),
        )

    def forward(self, x):
        out = self.model(x)
        if self.norm_reduce:
            out = torch.norm(out)

        return out


if __name__ == '__main__':
    model = MLP()
    x = torch.ones((50, 10))
    out = model(x)
    print(out.shape)
    num_params = sum(param.numel() for param in model.parameters())
    print(num_params)
