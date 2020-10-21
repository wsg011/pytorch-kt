import numpy as np

import torch
from torch.utils.data import Dataset


class AssistmentDataset(Dataset):
    def __init__(self, seqs, n_skill):
        super(AssistmentDataset, self).__init__()
        self.seqs = seqs
        self.n_skill = n_skill
    
    def __len__(self):
        return len(self.seqs)
    
    def encoding(self, seq):
        """
        encoding seq to training data format. 
        
        args:
            seq: user assisment
        egs:
            seq: [[358, 1], [359, 1], [360, 1], [361, 0]]
            output: 
             sequence: [358, 359, 360]
             target_id: [361]
             label: [0]
        """
        sequence = []
        for s in seq[:-1]:
            row = np.zeros(self.n_skill*2)

            skill_index = s[0]
            correct = s[1]

            row[skill_index - 1] = 1
            row[skill_index - 1 + self.n_skill] = correct
            
            sequence.append(row)
        
        target_id = seq[-1][0] - 1
        label = seq[-1][1]

        return np.array(sequence), np.array([target_id]), np.array([label])
    
    def __getitem__(self, index):
        seq = self.seqs[index]
        sequence, targe_id, label = self.encoding(seq)
        return sequence, targe_id, label


if __name__ == "__main__":
    dataset = AssistmentDataset("../data/", 381)