

from . import categorical_features
import pandas as  pd
from sklearn import linear_model




if __name__ == "__main__":

    df = pd.read_csv("../../input/kaggle_cat_data/train.csv")
    df_test = pd.read_csv("../../input/kaggle_cat_data/test.csv")
    sample = pd.read_csv("../../input/kaggle_cat_data/sample_submission.csv")

    train_len = len(df)

    df_test["target"] = -1

    #only concatened for kaggle. Since there might new labels in the test 
    #dataset. In the real world it sadom is the case.
    full_data = pd.concat([df, df_test])
    columns = [c for c in df.columns if c not in ["id", "target"]]
    
    cat_feats = categorical_features.CategoricalFeatures(full_data,
                                    categorical_features=columns,
                                    encoding_type="ohe",
                                    handle_na=True)
    
    full_data_transform = cat_feats.fit_transform()

    X = full_data_transform[:train_len, :]
    X_test = full_data_transform[train_len:, :]

    clf = linear_model.LogisticRegression()
    clf.fit(X, df.target.values)
    preds = clf.predict_proba(X_test)[:, 1]

    sample.loc[:, "target"] = preds
    sample.to_csv("../../input/kaggle_cat_data/example_cat_features_submission.csv", index = False)






    
