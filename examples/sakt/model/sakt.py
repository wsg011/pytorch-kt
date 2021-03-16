import numpy as np

import torch
import torch.nn as nn


def future_mask(seq_length):
    future_mask = np.triu(np.ones((seq_length, seq_length)), k=1).astype('bool')
    return torch.from_numpy(future_mask)


class FFN(nn.Module):
    def __init__(self, state_size=200):
        super(FFN, self).__init__()
        self.state_size = state_size

        self.lr1 = nn.Linear(state_size, state_size)
        self.relu = nn.ReLU()
        self.lr2 = nn.Linear(state_size, state_size)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.lr1(x)
        x = self.relu(x)
        x = self.lr2(x)
        return self.dropout(x)


class SAKTModel(nn.Module):
    def __init__(self, n_skill, max_seq=100, embed_dim=200):
        super(SAKTModel, self).__init__()
        self.n_skill = n_skill
        self.embed_dim = embed_dim

        self.embedding = nn.Embedding(2*n_skill+1, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq-1, embed_dim)
        self.e_embedding = nn.Embedding(n_skill+1, embed_dim)

        self.multi_att = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, dropout=0.2)

        self.dropout = nn.Dropout(0.2)
        self.layer_normal = nn.LayerNorm(embed_dim) 

        self.ffn = FFN(embed_dim)
        self.pred = nn.Linear(embed_dim, 1)
        self.sigmoid = nn.Sigmoid()

        self._reset_parameters()
    
    def _reset_parameters(self):
        r"""Initiate parameters in the model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, question_ids):
        device = x.device        
        # src_pad_mask = (x == 0)
        # tgt_pad_mask = (question_ids == 0)
        # mask = src_pad_mask & tgt_pad_mask

        x = self.embedding(x)
        pos_id = torch.arange(x.size(1)).unsqueeze(0).to(device)

        pos_x = self.pos_embedding(pos_id)
        x = x + pos_x

        e = self.e_embedding(question_ids)

        x = x.permute(1, 0, 2) # x: [bs, s_len, embed] => [s_len, bs, embed]
        e = e.permute(1, 0, 2)
        att_mask = future_mask(x.size(0)).to(device)
        att_output, att_weight = self.multi_att(e, x, x, attn_mask=att_mask)
        att_output = self.layer_normal(att_output + e)
        att_output = att_output.permute(1, 0, 2) # att_output: [s_len, bs, embed] => [bs, s_len, embed]
        # print(att_output.shape, att_weight.shape)
        x = self.ffn(att_output)
        # x = self.dropout(x)
        x = self.layer_normal(x + att_output)
        x = self.pred(x)

        return self.sigmoid(x.squeeze(-1)), att_weight


if __name__ == "__main__":
    q = torch.zeros((2, 9)).long()
    qa = torch.zeros((2, 9)).long()

    q_ = torch.randint(0, 100, size=(2, 6))
    qa_ = torch.randint(0, 2, size=(2, 6))

    q[:, -6:] = q_
    qa[:, -6:] = qa_

    x = q[:, :-1].clone()
    x += (qa[:, :-1] == 1) * 100
    e = q[:, -1].clone()
    question_ids = q[:, 1:].clone()

    model = SAKTModel(n_skill=100)
    print(x.shape, question_ids.shape)
    ouput, att_weight = model(x, question_ids)
    print(ouput.shape)
    print(att_weight)