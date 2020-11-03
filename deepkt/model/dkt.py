import torch
import torch.nn as nn
from torch.autograd import Variable


class DKTModel(nn.Module):
    def __init__(self, n_skill, hidden_size=200,
                 q_emb_dim=300, qa_emb_dim=100):
        super(DKTModel, self).__init__()
        self.n_skill = n_skill
        self.hidden_size = hidden_size

        self.q_emb = nn.Embedding(n_skill+1, q_emb_dim)
        self.qa_emb = nn.Embedding(2, qa_emb_dim)

        self.rnn = nn.LSTM(q_emb_dim+qa_emb_dim, hidden_size, batch_first=True, dropout=0.2)

        self.pred = nn.Linear(hidden_size, n_skill)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, q, qa):
        bs = q.size(0)
        device = q.device
        hidden = Variable(torch.zeros(1, bs, self.hidden_size)).to(device)
        cell = Variable(torch.zeros(1, bs, self.hidden_size)).to(device)

        x_q = self.q_emb(q)
        x_qa = self.qa_emb(qa)
        x = torch.cat([x_q, x_qa], dim=2)

        x, (h_n, c_n) = self.rnn(x, (hidden, cell)) # lstm output:[bs, seq_len, hidden] hidden [bs, hidden]

        x = self.pred(x[:, -1, :])

        return self.sigmoid(x)
    

if __name__ == "__main__":
    q = torch.randint(0, 100, size=(32, 100))
    qa = torch.randint(0, 2, size=(32, 100))

    model = DKTModel(n_skill=100)

    output = model(q, qa)
    print(output.shape)
    print(output)