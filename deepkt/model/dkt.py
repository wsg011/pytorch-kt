import torch
import torch.nn as nn
from torch.autograd import Variable


class DKTModel(nn.Module):
    def __init__(self, n_skill, hidden_size=200, emb_dim=300):

        super(DKTModel, self).__init__()
        self.n_skill = n_skill
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(2*n_skill+1, emb_dim)
        # self.qa_emb = nn.Embedding(2, qa_emb_dim)

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
    

if __name__ == "__main__":
    q = torch.randint(0, 100, size=(32, 100))
    qa = torch.randint(0, 2, size=(32, 100))

    x = q
    x += (qa == 1) * 100

    model = DKTModel(n_skill=100)

    output = model(x)
    print(output.shape)
    print(output)