
from . import cross_val2
if __name__ == "__main__":
    df = pd.read_csv("../../input/kaggle_cat_data/cat-in-the-dat-ii/train.csv")
    cv = cross_validation.CrossValidation(df, shuffle=True, target_cols=["attribute_ids"],
        problem_type="multilabel_classification", multilabel_delimiter=" ")

    df_split = cv.split()
    print(df_split.head())
    print(df_split.kfold.value_counts())
