from abc import ABC

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from random import sample

import torch
import matplotlib.pyplot as plt

from torch.utils.data import IterableDataset, DataLoader


class IrisDataset(IterableDataset, ABC):
    def __init__(self, x, y, idx):
        super(IrisDataset).__init__()
        self._x = x
        self._y = y
        self._idx = idx

    def __iter__(self):
        return zip(self._x[self._idx], self._y[self._idx])

    def __len__(self):
        return len(self._x)


def load_datasets(batch_size):
    df = pd.read_csv("data/iris.csv")

    # getting the input features from the dataset
    x = np.float32(df[["sepal_length", "sepal_width", "petal_length", "petal_width"]].values)
    # The LabelBinarizer converts a discrete label (i.e. setosa) into a binary numeric representation [1, 0, 0],
    # where each column is different for different labels
    # todo; Advanced: How would you implement a different strategy? Would that change the cost function?
    y = np.float32(LabelBinarizer().fit_transform(df["species"]))

    # select randomly 90% of the samples and place them into the training set
    train_idx = sample(range(len(x)), round(len(x) * 0.9))
    # the remaining samples (10%) will consist in the test set
    # todo; Advanced: how would you generated the validation set?
    test_idx = [x for x in range(len(x)) if x not in train_idx]

    train_ds = IrisDataset(x, y, train_idx)
    test_ds = IrisDataset(x, y, test_idx)

    return DataLoader(train_ds, batch_size=batch_size), DataLoader(test_ds, batch_size=batch_size)


def main():
    # the batch size is currently set to 4, but you can test with other batches to see if the results are changing
    train_dl, test_dl = load_datasets(batch_size=4)

    # Build a Sequential model, with a two dense linear level, a ReLU activation in the middle and a sigmoid activation function at the end
    # The input size has the match the number of features (4), while the output has to match the encoding layer (3)
    # The output of the first linear layer size has to match the input size of the second layer. In this case it is 8.
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 3),
        torch.nn.Sigmoid()
    )
    # What would be the most suitable loss function to use here?
    # Have a look at https://pytorch.org/docs/stable/nn.html#loss-functions and choose the right one.
    # Please note that the last activation function is already a sigmoid, so BCE Loss can be a good option.
    loss_fn = torch.nn.BCELoss()  # todo; replace this with a suitable loss function

    # Use the optim package to define an Optimizer that will update the weights of the model for us.
    # Have a look at https://pytorch.org/docs/stable/optim.html for more options
    # Adam is usually a safe option, and you have to pass model.parameters() to optimise correctly
    optimizer = torch.optim.Adam(model.parameters()) # todo; replace this with a suitable function

    # number of epochs to train on, you can check if this is decreasing
    num_epochs = 300
    avg_loss_list = []

    for epoch in range(num_epochs):
        total_loss = 0
        for x, y in train_dl:
            # Before the backward pass, use the optimizer object to zero all the
            # gradients for the Tensors it will update (which are the learnable weights
            # of the model)
            optimizer.zero_grad()

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            total_loss += loss

            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # Calling the step function on an Optimizer makes an update to its parameters
            optimizer.step()

        avg_loss = total_loss / len(train_dl)
        avg_loss_list.append(float(avg_loss))
        print(f"Epoch: {epoch}, Training loss: {avg_loss}")

    plt.plot([i + 1 for i in range(len(avg_loss_list))], avg_loss_list)
    plt.show()
    # todo; plot the average loss using matplotlib

    # todo; test the Precision, Recall, F1 of the model - you can use sklearn functions
    # these are the same functions you used in Tutorial 1


if __name__ == "__main__":
    main()
