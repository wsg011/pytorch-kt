import numpy as np
import pandas as pd

import torch
import torch.nn.utils.rnn as rnn_utils


def load_data(fn):
    dtype = {
        "skill_id": "int64",
        "user_id": "int64",
        "correct": "int64"
    }
    df = pd.read_csv(fn, dtype=dtype)

    if "skill_id" not in df.columns:
        raise KeyError(f"The column 'skill_id' was not found on {fn}")
    if "correct" not in df.columns:
        raise KeyError(f"The column 'correct' was not found on {fn}")
    if "user_id" not in df.columns:
        raise KeyError(f"The column 'user_id' was not found on {fn}")
    
    skills = df["skill_id"].unique()
    users = df["user_id"].unique()

    group_data = df.groupby('user_id').apply(lambda r: (
            r['skill_id'].values,
            r['correct'].values))

    # seqs = []
    # for user_id in group.index:
    #     data = group[user_id]
    #     x = data[0]
    #     y = data[1]

    #     seq = []
    #     for i in range(len(x)):
    #         seq.append([x[i], y[i]])
        
    #     if len(seq) > 101:
    #         seq = seq[-101:]
        
    #     if len(seq) < 2:
    #         continue
        
    #     seqs.append(seq)
    return group_data, skills, users


def collate_fn(data):
    qs = []
    qas = []
    target_ids = []
    labels = []
    batch_size = len(data)
    for i in range(batch_size):
        q = torch.from_numpy(data[i][0])
        qa = torch.from_numpy(data[i][1])
        target_id = data[i][2]
        label = data[i][3]

        qs.append(q)
        qas.append(qa)
        target_ids.append(target_id)
        labels.append(label)
    
    qs = rnn_utils.pad_sequence(qs, batch_first=True, padding_value=0)
    qas = rnn_utils.pad_sequence(qas, batch_first=True, padding_value=0)

    target_ids = np.array(target_ids)
    target_ids = torch.from_numpy(target_ids)

    labels = np.array(labels)
    labels = torch.from_numpy(labels)

    return (qs, qas, target_ids, labels)