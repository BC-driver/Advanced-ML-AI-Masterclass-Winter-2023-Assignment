import pandas as pd
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, accuracy_score, cohen_kappa_score


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
    tp = 
    result =
    # todo; implement this and return the correct value
    return result


def compute_precision_score(y, yp):
    # todo; implement this and return the correct value
    return 0.0


def compute_f1_score(y, yp):
    # todo; implement this and return the correct value
    return 0.0


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
    # todo; implement this and return the correct value

    return 0.0


def compute_vote_agreement(row):
    result = row.mode()[0]
    return result


def compute_metrics(y, yp):
    m = compute_confusion_matrix(y, yp)
    display_confusion_matrix(m)

    # print(f"   Recall: {compute_recall_score(y, yp):.4}")
    # print(f"Precision: {compute_precision_score(y, yp):.4}")
    # print(f"       F1: {compute_f1_score(y, yp):.4}")
    # print(f" Accuracy: {compute_accuracy_score(y, yp):.4}")
    # print(f"        K: {compute_cohen_kappa_score(y, yp):.4}")

    print(
        f"   Recall: {compute_recall_score(y, yp):.4} The value computed by Sklearn is: {recall_score(y, yp, average='macro'):.4}")
    print(
        f"Precision: {compute_precision_score(y, yp):.4} The value computed by Sklearn is: {precision_score(y, yp, average='macro'):.4}")
    print(
        f"       F1: {compute_f1_score(y, yp):.4}. The value computed by Sklearn is: {f1_score(y, yp, average='macro'):.4}")
    print(f" Accuracy: {compute_accuracy_score(y, yp):.4} The value computed by Sklearn is: {accuracy_score(y, yp):.4}")
    print(
        f"        K: {compute_cohen_kappa_score(y, yp):.4} The value computed by Sklearn is: {cohen_kappa_score(y, yp):.4}")


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
