import pandas as pd 
from sklearn import model_selection



"""
Types of cross validation:
K-fold
Stratified K_fold
Multilabel classification
Regression Classification
Holdout based validation

In the sample dataset, it is binary classification

Stratified classification:
It splits the data but keep the ratio of the positive
and negative data same in each fold. For example:
if you have 20% positive sample on the training then you will 
have 20% on the validation set too.

"""

class CrossValidation:
    def __init__(self, 
                df,
                target_cols,
                shuffle,
                problem_type = "binary_classification",
                multilabel_delimiter = ",",
                num_folds=5,
                random_state=42
    ):

        self.dataframe = df 
        self.target_cols = target_cols 
        self.num_targets = len(target_cols)
        self.problem_type = problem_type
        self.multilabel_delimiter = multilabel_delimiter
        self.random_state = random_state
        self.shuffle = shuffle
        self.num_folds = num_folds

        if self.shuffle is True:
            self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)

        #create a new col and initialize with the 
        self.dataframe["kfold"] = -1 

    #Remarks: Kfold regression: only X split whereas, KFold classification both X and y split.
    def split(self):
        if self.problem_type in ("binary_classification", "multiclass_classification"):
            if self.num_targets != 1:
                raise Exception("Invalid number of target for this problem type")

            target = self.target_cols[0]
            #count the unique values
            unique_values = self.dataframe[target].nunique()
            if unique_values == 1:
                raise Exception ("only one value is found")
            elif unique_values >1:
            #It is good idea to have stratified classification here to ensure that the ratio of postive
            # to negative value stays the same.
                kf = model_selection.StratifiedKFold(n_splits=self.num_folds,
                                                    shuffle=False)
                
                for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe,
                                                                    y=self.dataframe[target].values,
                                                                     )):
                    self.dataframe.loc[val_idx, 'kfold'] = fold

        elif self.problem_type in ("single_col_regression". "multi_col_regression"):
            if num_targets != 1 and self.problem_type == "single_col_regression":
                raise Exception("Invalid num of traget for this type of problem")
            if self.num_targets <2  and self.problem_type =="multiclass_classification":
                raise Exception("Invalid num of traget for this type of problem")

                kf =model_selection.KFold(n_splits=self.num_folds)

                for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe)):
                    self.dataframe[val_idx, "kfold"] = fold


        elif self.problem_type.startswith("holdout_"):
            holdout_percentage = int(self.problem_type.split(" ")[1])
            num_holdout_samples = int(len(self.dataframe)*holdout_percentage/100)
            # TODO: investigate
            self.dataframe.loc[:len(self.dataframe) - num_holdout_samples, "kfold"] =0
            self.dataframe.loc[len(self.dataframe) - num_holdout_samples:, "kfold"] = 1

        elif self.problem_type == "multilabel_classification":
            #Each sample can have more than one labels. 

            if self.num_targets != 1:
                raise Exception ("Invalid num of traget for this type of problem ")
            targets = self.dataframe[self.target_cols[0]].apply(lambda x: len(str(x).split(self.multilabel_delimiter)))
            kf = model_selection.StratifiedKFold(n_splits=self.num_folds)
            for fold, (train_idx, val_idx) in enumerate(kf.split(X= self.dataframe, y = targets)):
                self.dataframe.loc[val_idx, "kfold"] = fold

        # TODO Add more type of prom type here.

        else:
            raise Exception ("problem_type not understood")

        return self.dataframe


if __name__ == "__main__":
    df == pd.read_csv("../")

















