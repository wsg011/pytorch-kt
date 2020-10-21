#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author  :   wsg011
@Email   :   wsg20110828@163.com
@Time    :   2020/10/20 15:27:46
@Desc    :   
'''
import os
import argparse
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("--data", default="ASSISTments2019", type=str)
parser.add_argument("--path", default="../data/ASSISTments2009/skill_builder_data_corrected.csv", type=str)
args = parser.parse_args()


if __name__ == "__main__":
    if args.data == "ASSISTments2019":
        df = pd.read_csv(args.path)

        data = pd.DataFrame()
        data["user_id"] = df["user_id"]
        data["skill_id"] = df["skill_id"]
        data["correct"] = df["correct"]
        data = data.dropna()

        print(df.columns)

    # save
    save_file = os.path.join("../data", "{}.csv".format(args.data))
    data.to_csv(save_file, index=False)