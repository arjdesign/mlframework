import os
import pandas as pd 
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import joblib
import numpy as np
from . import dispatcher

def predict(test_data_path, model_type, model_path):
    df = pd.read_csv(test_data_path)
    test_idx = df["id"].values
    predictions = None

    for FOLD in range(1):
        df = pd.read_csv(test_data_path)
        encoders = joblib.load(os.path.join(model_path, 
                f"{model_type}_{FOLD}_labelencoder.pkl"))
        cols = joblib.load(os.path.join(model_path, 
                f"{model_type}_{FOLD}_columns.pkl"))

        for c in encoders:
            lbl = encoders[c]
            df.loc[:,c] = df.loc[:,c].astype(str).fillna("NONE")
            df.loc[:,c] = lbl.transform(df[c].values.tolist())
        
        clf = joblib.load(os.path.join(model_path, f"{model_type}_{FOLD}.pkl"))

        df = df[cols]
        #prediction
        preds = clf.predict_proba(df)[:,1]

        #average the prediction result

        if FOLD == 0:
            predictions = preds
        else:
            predictions += preds

    predictions /= 5
    
    #stack on columns to create df
    sub = pd.DataFrame(np.column_stack((test_idx, predictions)), 
                columns=["id", "target"])
    return sub


if __name__ == "__main__":
    submission = predict(test_data_path="input/test.csv",
                        model_type = "xgboost",
                        model_path ="models/")
    #sanity check
    submission.loc[:, "id"] = submission.loc[:, "id"].astype(int)
    submission.to_csv(f"models/xgb_submission.csv", index=False)




