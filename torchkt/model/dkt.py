#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author  :   wsg011
@Email   :   wsg20110828@163.com
@Time    :   2021/03/15 17:58:19
@Desc    :   Deep Neural Network for Knowledge Tracing
'''
import torch
import torch.nn as nn
from torch.autograd import Variable

from .base_model import BaseModel


class DKTModel(BaseModel):
    def __init__(self, n_skill, hidden_size=100, emb_dim=100):

        super(DKTModel, self).__init__()
        self.n_skill = n_skill
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(2*n_skill+1, emb_dim)

        self.lstm = nn.LSTM(emb_dim, hidden_size, batch_first=True, dropout=0.2)

        self.pred = nn.Linear(hidden_size, n_skill)
    
    def forward(self, x):
        bs = x.size(0)
        device = x.device
        hidden = Variable(torch.zeros(1, bs, self.hidden_size)).to(device)
        cell = Variable(torch.zeros(1, bs, self.hidden_size)).to(device)

        x = self.embedding(x)

        x, _ = self.lstm(x, (hidden, cell)) # lstm output:[bs, seq_len, hidden] hidden [bs, hidden]
        x = self.pred(x[:, -1, :])

        return x