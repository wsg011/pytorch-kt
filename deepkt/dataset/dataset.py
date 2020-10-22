import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset


class AssistmentDataset(Dataset):
    def __init__(self, data, skills, max_seq=100):
        super(AssistmentDataset, self).__init__()
        self.skills = skills
        self.n_skill = len(skills)
        self.user_ids = data.index
        self.max_seq = max_seq
        self.data = data

    def __len__(self):
        return len(self.data)
    
    # def encoding(self, seq):
    #     """
    #     encoding seq to training data format. 
        
    #     args:
    #         seq: user assisment
    #     egs:
    #         seq: [[358, 1], [359, 1], [360, 1], [361, 0]]
    #         output: 
    #          sequence: [358, 359, 360]
    #          target_id: [361]
    #          label: [0]
    #     """
    #     sequence = []
    #     for s in seq[:-1]:
    #         row = np.zeros(self.n_skill*2)

    #         skill_index = s[0]
    #         correct = s[1]

    #         row[skill_index - 1] = 1
    #         row[skill_index - 1 + self.n_skill] = correct
            
    #         sequence.append(row)
        
    #     target_id = seq[-1][0] - 1
    #     label = seq[-1][1]

    #     return np.array(sequence), np.array([target_id]), np.array([label])

    def __getitem__(self, index):
        user_id = self.user_ids[index]

        q, qa = self.data[user_id]

        if len(q) > self.max_seq:
            q = q[-self.max_seq:]
            qa = qa[-self.max_seq:]

        target_id = q[-1]
        label = qa[-1]

        q = q[:-1]
        qa = qa[:-1]

        return q, qa, [target_id], [label]


if __name__ == "__main__":
    dataset = AssistmentDataset("../../data/ASSISTments2009.csv")

    q, qa, target_id, label = dataset.__getitem__(10)
    print(q.shape, qa.shape, target_id, label)