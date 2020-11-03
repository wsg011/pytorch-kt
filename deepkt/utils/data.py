import numpy as np
import pandas as pd


def load_data(path):
    """
    load dataset skills and users

    args:
        path: dataset path
    """
    skill_df = pd.read_csv(path+"/skills.csv")
    user_df = pd.read_csv(path+"/users.csv")

    skills = skill_df["skill_id"].unique()
    skills = sorted(skills)

    users = user_df["user_id"].unique()
    users = sorted(users)

    return skills, users