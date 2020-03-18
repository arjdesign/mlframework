import os
import pandas as pd 
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import joblib

from . import dispatcher

TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")
FOLD = int(os.environ.get("FOLD"))
MODEL = os.environ.get("MODEL")

FOLD_MAPPING = {
    0: [1,2,3,4],
    1: [0,2,3,4],
    2: [0,1,3,4],
    3: [0,1,2,4],
    4: [0,1,2,3]
}


if __name__ = "__main__":
    df = pd.read_csv(TRAINING_DATA)
    df_test = pd.read_csv(TEST_DATA)

    train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))].rest_index(drop=True)
    valid_df = df[df.kfold = FOLD].rest_index(drop = True)

    ytrain = train_df.target.values
    yvalid = valid_df.target.values

    train_df = train_df.drop(["id", "target", "kfold"], axis =1)
    valid_df = valid_df.drop(["id", "target", "kfold"], axis =1)

    #match columns of train_df and valid_df
    valid_df = valid_df[train_df.columns]

    label_encoders = {}
    for c in train_df.columns:
        lbl = preprocessing.LabelEncoder()
        






    

