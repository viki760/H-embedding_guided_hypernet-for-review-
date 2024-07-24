import torch
import torch.nn as nn

class EmbDecoder(nn.Module):
    def __init__(self, hidden_dim, emb_dim):
        super(EmbDecoder, self).__init__()
        self.fc = nn.Linear(hidden_dim, emb_dim)  # 从input_features到32维输出

    def forward(self, h):
        output = self.fc(h)
        return output

