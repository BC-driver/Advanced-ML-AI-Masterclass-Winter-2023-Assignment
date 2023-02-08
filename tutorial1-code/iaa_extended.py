import pandas as pd


def compute_confusion_matrix(y, yp):
    labels = pd.DataFrame(y)[0].unique()
    result = pd.DataFrame(columns=labels, index=labels)
    result.loc[:, :] = 0
    for i in range(len(y)):
        result.loc[y[i], yp[i]] += 1
    return result


def display_confusion_matrix(m):
    print(m)


def compute_recall_score(y, yp):
    cm = compute_confusion_matrix(y, yp)
    tp = pd.Series(data=[cm[idx][idx] for idx, row in cm.iterrows()])
    recallScores = tp / cm.apply(lambda x: x.sum(), axis=1).reset_index(drop=True)
    result = recallScores.sum() / len(cm)
    return result


def compute_precision_score(y, yp):
    cm = compute_confusion_matrix(y, yp)
    tp = pd.Series(data=[cm[idx][idx] for idx, row in cm.iterrows()])
    precisionScores = tp / cm.apply(lambda x: x.sum(), axis=0).reset_index(drop=True)
    result = precisionScores.sum() / len(cm)
    return result


def compute_f1_score(y, yp):
    cm = compute_confusion_matrix(y, yp)
    tp = pd.Series(data=[cm[idx][idx] for idx, row in cm.iterrows()])
    f1Scores = (tp * 2) / (cm.apply(lambda x: x.sum(), axis=0).reset_index(drop=True) +
                           cm.apply(lambda x: x.sum(), axis=1).reset_index(drop=True))
    result = f1Scores.sum() / len(cm)
    return result


def compute_accuracy_score(y, yp):
    cm = compute_confusion_matrix(y, yp)
    correctCnt = 0
    total = len(y)
    for idx, row in cm.iterrows():
        correctCnt += row[idx]
    return correctCnt / total


def compute_cohen_kappa_score(y, yp):
    cm = compute_confusion_matrix(y, yp)
    po = compute_accuracy_score(y, yp)
    pe = cm.apply(lambda x: x.sum(), axis=1).transpose().dot(cm.apply(lambda x: x.sum(), axis=0))
    pe /= (len(y) ** 2)
    return (po - pe) / (1 - pe)


def compute_fleiss_kappa_score(df):
    # Fleiss' Kappa score is an extension of Cohen's Kappa for more than 2 annotators
    N = df.shape[0]
    n = df.shape[1]
    frequency = df.apply(lambda x: x.value_counts(), axis=1).fillna(0)
    pj = frequency.apply(lambda x: x.sum(), axis=0) / (N * n)
    pe = (pj ** 2).sum()
    pi = frequency.apply(lambda x: (x ** 2).sum() - n, axis=1) / (n * (n - 1))
    po = pi.sum() / N
    result = (po - pe) / (1 - pe)
    return result


def compute_vote_agreement(row):
    result = row.mode()[0]
    return result


def compute_metrics(y, yp):
    m = compute_confusion_matrix(y, yp)
    display_confusion_matrix(m)

    print(f"   Recall: {compute_recall_score(y, yp):.4}")
    print(f"Precision: {compute_precision_score(y, yp):.4}")
    print(f"       F1: {compute_f1_score(y, yp):.4}")
    print(f" Accuracy: {compute_accuracy_score(y, yp):.4}")
    print(f"        K: {compute_cohen_kappa_score(y, yp):.4}")


def main():
    df = pd.read_csv("data/coarse_discourse_dataset.csv")

    # compute the agreed label via majority voting
    df["majority_label"] = df.apply(compute_vote_agreement, axis=1)

    print(df.head())
    print()

    print("--- Comparing annotator 1 vs annotator 2")
    compute_metrics(df.annot1.values, df.annot2.values)
    print("--- Comparing majority vs annotator 1")
    compute_metrics(df.majority_label.values, df.annot1.values)
    print("--- Comparing majority vs annotator 2")
    compute_metrics(df.majority_label.values, df.annot2.values)
    print("--- Comparing majority vs annotator 3")
    compute_metrics(df.majority_label.values, df.annot3.values)

    print(f" Fleiss Kappa: {compute_fleiss_kappa_score(df[['annot1', 'annot2', 'annot3']]):.4}")


if __name__ == "__main__":
    main()
