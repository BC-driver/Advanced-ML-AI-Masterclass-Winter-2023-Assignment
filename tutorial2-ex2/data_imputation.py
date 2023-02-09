import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.impute import KNNImputer


def kNN(df, i, j, k):
    targetColumn = df[j]
    restColumn = df.drop(j, axis=1)
    targetRow = restColumn.loc[i, :]
    distance = restColumn.apply(lambda x: ((x - targetRow) ** 2).sum() ** 0.5, axis=1)
    distance = distance.sort_values()
    neighborIdx = list(distance[1: k].index)
    predict = targetColumn[neighborIdx].mean()
    return predict


def missing_values(df):
    # todo; calculate the data availability for every column
    # missingCnt = df.isna().sum()
    # labelRank = list(missingCnt.sort_values().index)

    # todo; order columns by data availability
    # df = df[labelRank]

    # todo; use the dataset mean and variance to find out missing value
    dfNormalFilling = df.apply(lambda x: x.fillna(np.random.normal(x.mean(), x.var())))

    # todo; use a KNN to find out missing value
    # For beginners, you can use the sklearn knn data imputation implementation: sklearn.impute.KNNImputer
    # Advanced students can implement their own KNN data imputation pipeline
    imputer = KNNImputer(n_neighbors=5)
    dfSkkNNFilling = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    nanPos = df.isnull().stack()[lambda x: x].index.tolist()
    dfkNNFilling = df.fillna(0)
    for idx, label in nanPos:
        dfkNNFilling.loc[idx, label] = kNN(dfkNNFilling, idx, label, 5)

    # todo; compare both strategies with the original dataset

    return dfSkkNNFilling


def main():
    # if you prefer to work on normalization only, you can work with the original dataset
    df = pd.read_csv("data/iris_numeric_dataset.missing.csv")
    df_original = pd.read_csv("data/iris_numeric_dataset.original.csv")

    # print the missing numbers
    print("Missing numbers:")
    print(df.isna().sum())
    print(df)

    df = missing_values(df)

    # calculate the Mean Squared Error between the predicted missing values and original data
    mse_error = df.fillna(0).apply(lambda column: mean_squared_error(df_original[column.name], column.values),
                                   axis=0).mean()
    print(f"Mean Squared Error between the missing data (corrected) and original is {mse_error}")
    # origin        :2.358186666666666
    # Sklearn-kNN   :0.0196992
    # mykNN         :0.16340688151041668
    # random normal :0.45921051698729193

if __name__ == '__main__':
    main()
