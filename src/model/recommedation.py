import torch
import torch.nn as nn
import numpy as np

class RecModel(nn.Module):
    def __init__(self, user_tower, item_tower):
        super().__init__()
        self.user_tower = user_tower
        self.item_tower = item_tower
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, batch):
        u_vec = self.user_tower(batch)
        v_vec = self.item_tower(batch)
        
        # Dot Product
        logits = torch.matmul(u_vec, v_vec.T) * self.logit_scale.exp()
        return logits