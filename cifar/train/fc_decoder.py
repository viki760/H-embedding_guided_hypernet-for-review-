import torch
import torch.nn as nn

class EmbDecoder(nn.Module):
    def __init__(self, hidden_dim, emb_dim, mid_dim=512):
        super(EmbDecoder, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, mid_dim)  # 从input_features到32维输出
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(mid_dim, emb_dim)


    def forward(self, h):
        h = self.relu(self.fc1(h))
        output = self.fc2(h)
        return output

