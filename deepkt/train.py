#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author  :   wsg011
@Email   :   wsg20110828@163.com
@Time    :   2020/10/20 16:08:36
@Desc    :   
'''
import os
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader

from dataset.dataset import AssistmentDataset
from model.dkt import DKTModel
from utils import data


logger = logging.Logger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--path", default="../data", help="dataset file path")
parser.add_argument("--fn", default="ASSISTments2009.csv", help="training dataset filename")
parser.add_argument("--epochs", default=20, help="training epoch numbers")
args = parser.parse_args()


def train(model, train_iterator, optim, criterion, device="cpu"):
    model.train()

    train_loss = []
    num_corrects = 0
    num_total = 0
    labels = []
    outs = []

    tbar = tqdm(train_iterator)
    for item in tbar:
        q = item[0].to(device).long()
        qa = item[1].to(device).long()
        target_id = item[2].to(device).long()
        label = item[3].to(device).float()
        # seq = seq.to(device)
        # target_id = seq.to(device)
        # label = label.to(device)

        optim.zero_grad()
        output = model(q, qa)

        output = torch.gather(output, -1, target_id-1)
        #pred = pred[:, target_id]
        #pred = pred.squeeze(dim=-1)

        pred = (torch.sigmoid(output) >= 0.5).long()
        
        loss = criterion(output, label)
        loss.backward()
        optim.step()

        train_loss.append(loss.item())
        num_corrects += (pred == label).sum().item()
        num_total += len(label)

        labels.extend(label.squeeze(-1).data.cpu().numpy())
        outs.extend(output.squeeze(-1).data.cpu().numpy())

        tbar.set_description('loss - {:.4f}'.format(loss))


    acc = num_corrects / num_total
    auc = roc_auc_score(labels, outs)
    loss = np.mean(train_loss)

    return loss, acc, auc


def validation(model, val_iterator, criterion, device):
    model.eval()

    val_loss = []
    num_corrects = 0
    num_total = 0
    labels = []
    outs = []

    tbar = tqdm(val_iterator)
    for item in tbar:
        q = item[0].to(device).long()
        qa = item[1].to(device).long()
        target_id = item[2].to(device).long()
        label = item[3].to(device).float()

        with torch.no_grad():
            output = model(q, qa)
    
        output = torch.gather(output, -1, target_id-1)
        pred = (torch.sigmoid(output) >= 0.5).long()
        loss = criterion(output, label)

        val_loss.append(loss.item())
        num_corrects += (pred == label).sum().item()
        num_total += len(label)

        labels.extend(label.squeeze(-1).data.cpu().numpy())
        outs.extend(output.squeeze(-1).data.cpu().numpy())

        tbar.set_description('loss - {:.4f}'.format(loss))

    acc = num_corrects / num_total
    auc = roc_auc_score(labels, outs)
    loss = np.mean(val_loss)

    return loss, acc, auc


if __name__ == "__main__":
    fn = os.path.join(args.path, args.fn)

    seqs, skills, user_ids = data.load_data(fn)
    train_data, val_data = train_test_split(seqs, test_size=0.2)

    n_skill = np.max(skills)

    print("dataset has seq {}, number skill {}, number user {}.".format(len(seqs), n_skill, len(user_ids)))


    train_dataset = AssistmentDataset(train_data, skills)
    val_dataset = AssistmentDataset(val_data, skills)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=data.collate_fn)
    
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=data.collate_fn)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = DKTModel(n_skill)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.99, weight_decay=0.005)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()


    model.to(device)
    criterion.to(device)

    epochs = args.epochs
    for epoch in range(epochs):
        loss, acc, auc = train(model, train_dataloader, optimizer, criterion, device)
        print("epoch - {} train_loss - {:.2f} acc - {:.2f} auc - {:.2f}".format(epoch, loss, acc, auc))

        val_loss, val_acc, val_auc = validation(model, val_dataloader, criterion, device)
        print("epoch - {} vall_loss - {:.2f} acc - {:.2f} auc - {:.2f}".format(epoch, val_loss, val_acc, val_auc))



