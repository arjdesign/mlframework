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

# TO DO: refactor it as train()
if __name__ == "__main__":
    df = pd.read_csv(TRAINING_DATA)
    df_test = pd.read_csv(TEST_DATA)

    train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))].reset_index(drop=True)
    valid_df = df[df.kfold == FOLD].reset_index(drop = True)

    ytrain = train_df.target.values
    yvalid = valid_df.target.values

    train_df = train_df.drop(["id", "target", "kfold"], axis =1)
    valid_df = valid_df.drop(["id", "target", "kfold"], axis =1)

    #match columns of train_df and valid_df
    valid_df = valid_df[train_df.columns]

    label_encoders = {}
    for c in train_df.columns:
        lbl = preprocessing.LabelEncoder()
        train_df.loc[:,c] = train_df.loc[:,c].astype(str).fillna("NONE")
        valid_df.loc[:,c] = valid_df.loc[:,c].astype(str).fillna("NONE")
        df_test.loc[:,c] = df_test.loc[:,c].astype(str).fillna("NONE") 
        #fit

        #Label encoder will fail if you have new labels. so, proper way of doing this would be 
        # to add a  "new" label to training labels and fitting the encoder. this way,
        #  all new labels in validation or test set will become "new". im not sure how it 
        # will perform. what im doing is a kind of semi-supervised learning. 
        # it may cause data leakage if you do target encoding or similar things

        lbl.fit(train_df[c].values.tolist()+
                valid_df[c].values.tolist()+
                df_test[c].values.tolist())

        train_df.loc[:,c] = lbl.transform(train_df[c].values.tolist())
        valid_df.loc[:,c] = lbl.transform(valid_df[c].values.tolist())
        label_encoders[c]= lbl


    #data is preprocessed and ready to train

    clf = dispatcher.MODELS[MODEL]
    clf.fit(train_df, ytrain)
    #get probability
    preds = clf.predict_proba(valid_df)[:,1]
    print(f"ROC_AUC accuracy score: {metrics.roc_auc_score(yvalid, preds)}")


    joblib.dump(label_encoders, f"models/{MODEL}_{FOLD}_labelencoder.pkl")
    joblib.dump(clf, f"models/{MODEL}_{FOLD}.pkl")
    joblib.dump(train_df.columns, f"models/{MODEL}_{FOLD}_columns.pkl")








    

