import pandas as pd
import numpy as np


def encode_unique_values(unique_values):
    return {v: i for i, v in enumerate(unique_values)}


# this is not the optimised version of this algorithm
# the current version was chosen because it is easier to understand
def count_values_per_question(n, k, df, unique_values):
    n_matrix = np.zeros((n, k))

    # iterate over all rows in the dataset
    for i in range(n):
        # iterate over all values in a row
        for v in df.iloc[i].values:
            # convert the value to the index according to the unique values convention
            j = unique_values[v]
            # increment the unique value
            n_matrix[i, j] = n_matrix[i, j] + 1

    return n_matrix


def Scompute_fleiss_kappa_score(df):
    # Fleiss' Kappa score is an extension of Cohen's Kappa for more than 2 annotators

    unique_values = encode_unique_values(set(df.annot1.unique()).union(set(df.annot2.unique())).union(set(df.annot3.unique())))
    k = len(unique_values)
    # n is the number of questions
    n = df.shape[0]
    # r is the number of raters (annotators)
    r = df.shape[1]
    # m is the matrix of counts of unique values per question
    m = count_values_per_question(n, k, df, unique_values)

    # pj - the proportion of all assignments which were to the j-th category
    pj = np.sum(m, axis=0) / (n * r)
    # p - the extent to which raters agree for the i-th subject
    p = (np.sum(m * m, axis=1) - r) / (r * (r - 1))
    # average of the p value
    p_m = np.sum(p) / n
    # sum of the squares of the pj values
    p_e = np.sum(pj * pj)

    # Fleiss' Kappa, as per formula
    return (p_m - p_e) / (1 - p_e)


def main():
    df = pd.read_csv("data/coarse_discourse_dataset.csv")
    print(f" Fleiss Kappa: {Scompute_fleiss_kappa_score(df[['annot1', 'annot2', 'annot3']]):.4}")


if __name__ == "__main__":
    main()
