import torch
import torch.nn as nn


class DKTModel(nn.Module):
    def __init__(self, n_skill, max_seq=100, hidden_size=100):
        super(DKTModel, self).__init__()
        self.n_skill = n_skill
        self.max_seq = max_seq

        self.rnn = nn.LSTM(n_skill*2, hidden_size, batch_first=True, dropout=0.2)
        self.pred = nn.Linear(hidden_size, n_skill)
    
    def forward(self, x):
        x, _ = self.rnn(x) # lstm output:[bs, seq_len, hidden] hidden [bs, hidden]
        x = self.pred(x[:, -1, :])

        return x
    

if __name__ == "__main__":
    inputs = torch.randint(0, 2, size=(32, 100, 200))

    model = DKTModel(n_skill=100)

    output = model(inputs.float())
    print(inputs)
    print(output.shape)
    print(output)