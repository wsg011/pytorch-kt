import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def fit(self, x_train, y_train, x_val=None, y_val=None):
        return None