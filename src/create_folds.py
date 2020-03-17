import pandas as pd 
from sklearn import model_selection


if __name__ == "__main__":
    df = pd.read_csv("input/train.csv")
    #make a fake column assign a value -1
    df["kfold"] = -1

    #shuffle the data
    df.sample(frac=1).reset_index(drop=True)
    kf = model_selection.StratifiedKFold(n_splits =5, shuffle=False, random_state =32)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X=df, y = df.target.values)):
        print(len(train_idx), len(val_idx))
        #assign k-folds`
        df.loc[val_idx, "kfold"] = fold
    
    #save the file
    df.to_csv("input/train_folds.csv", index=False)




