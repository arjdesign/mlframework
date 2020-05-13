import pandas as pd
from sklearn import model_selection


"""
Types of cross validation:
K-fold
Stratified K_fold
Multilabel classification
Regression - single_col_regression, multi_col_regression
Holdout based validation

In the sample dataset, it is binary classification

Stratified classification:
It splits the data but keep the ratio of the positive
and negative data same in each fold. For example:
if you have 20% positive sample on the training then you will 
have 20% on the validation set too.
utrerrrr
"""


class CrossValidation:
    def __init__(self,
                 df,
                 target_cols,
                 shuffle,
                 problem_type="binary_classification",
                 multilabel_delimiter=",",
                 num_folds=5,
                 random_state=42
                 ):

        self.dataframe = df
        # list of col names
        self.target_cols = target_cols
        self.num_targets = len(target_cols)
        self.problem_type = problem_type
        self.multilabel_delimiter = multilabel_delimiter
        self.random_state = random_state
        self.shuffle = shuffle
        self.num_folds = num_folds

        if self.shuffle is True:
            self.dataframe = self.dataframe.sample(
                frac=1).reset_index(drop=True)

        self.dataframe["kfold"] = -1
        # ----------------------------------------------------------------------------------------
        #  Remarks: Kfold : only X split whereas, StratifiedKFold needs both X and y split.
        #  It is good idea to have stratified classification here to ensure that the ratio of postive
        #  to negative value stays the same.
        # kf.split() is a generator that generates train_index and valid index
        # ----------------------------------------------------------------------------------------

    def split(self):
        if self.problem_type in ("binary_classification", "multiclass_classification"):
            if self.num_targets != 1:
                raise Exception(
                    "Invalid number of target for this problem type")
            target = self.target_cols[0]
            unique_values = self.dataframe[target].nunique()
            if unique_values == 1:
                raise Exception("only one value is found")
            elif unique_values > 1:
                kf = model_selection.StratifiedKFold(n_splits=self.num_folds,
                                                     shuffle=False)
                for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe,
                                                                     y=self.dataframe[target].values,
                                                                     )):
                    self.dataframe.loc[val_idx, 'kfold'] = fold

        # ----------------------------------------------------------------------------------------
        elif self.problem_type in ("single_col_regression", "multi_col_regression"):
            if self.num_targets != 1 and self.problem_type == "single_col_regression":
                raise Exception(
                    "Invalid num of traget for this type of problem")
            if self.num_targets < 2 and self.problem_type == "multiclass_classification":
                raise Exception(
                    "Invalid num of traget for this type of problem")

                kf = model_selection.KFold(n_splits=self.num_folds)

                for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe)):
                    self.dataframe[val_idx, "kfold"] = fold

        # ----------------------------------------------------------------------------------------------
        # Holdout is very important when you have time series data.
        # In timeseries if you use k-fold, you will be using data from future timestamp. It is
        # not a good idea. It will overfit your model and you will get really nice validation score
        # but it doesnot mean anything.
        # In case of time series data, set shuffle = False

        # When you have millions of dataset, then you can use holdout set. It is very
        # expensive to do 5/10 fold cross validation with such a large dataset. The best way is to
        # use holdout set in such condition.

        # In this case you really do not care about if it is reagression or classification.

        # EG: holdout_5, holdout_10
        # --------------------------------------------------------------------------------------
        elif self.problem_type.startswith("holdout_"):
            holdout_percentage = int(self.problem_type.split(" ")[1])
            num_holdout_samples = int(
                len(self.dataframe)*holdout_percentage/100)

            self.dataframe.loc[:len(self.dataframe) -
                               num_holdout_samples, "kfold"] = 0

            # this is holdout set
            self.dataframe.loc[len(self.dataframe) -
                               num_holdout_samples:, "kfold"] = 1

        # --------------------------------------------------------------------------------------
        # Each sample can have more than one labels.
        # Example: you have an image and it has multiple objects in it.
        # A single target column should have multiple labls with a delimiter. If you have multiple columns,
        # combine the colums with comma diliminator.

        # Example.
        # """
        # id  target
        # 1,  32, 60
        # 2,  50, 32, 15, 2
        # 3,  1,
        # """

        # create folds based on the counts of number of classes.
        # --------------------------------------------------------------------------------------
        elif self.problem_type == "multilabel_classification":

            if self.num_targets != 1:
                raise Exception(
                    "Invalid num of traget for this type of problem ")
            #TODO: investigate
            targets = self.dataframe[self.target_cols[0]].apply(
                lambda x: len(str(x).split(self.multilabel_delimiter)))
            kf = model_selection.StratifiedKFold(n_splits=self.num_folds)
            for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe, y=targets)):
                self.dataframe.loc[val_idx, "kfold"] = fold
        # -------------------------------------------------------------------------------------
        # TODO Add more problem_types here.
        # --------------------------------------------------------------------------------------
        else:
            raise Exception("problem_type not understood")

        return self.dataframe

        # TODO: split in such a way that you keep the distribution of the values similar. for both single column and multiple column.
        #


if __name__ == "__main__":
    df == pd.read_csv("../")

