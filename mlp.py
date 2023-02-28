import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from random import sample
import torch.optim as optim
from matplotlib.colors import LinearSegmentedColormap
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt

class MLPS(nn.Module):
    def __init__(self):
        super().__init__()
        self.FCBegin = nn.Linear(2, 10)
        self.FCEnd = nn.Linear(10, 4)

    def forward(self, x):
        x = self.FCBegin(x)
        x = F.relu(x)

        x = self.FCEnd(x)

        return x

class MLPM(nn.Module):
    def __init__(self):
        super().__init__()
        self.FCBegin = nn.Linear(2, 5)
        self.FC2 = nn.Linear(5, 10)
        self.FC3 = nn.Linear(10, 10)
        self.FCEnd = nn.Linear(10, 4)

    def forward(self, x):
        x = self.FCBegin(x)
        x = F.relu(x)

        x = self.FC2(x)
        x = F.relu(x)

        x = self.FC3(x)
        x = F.relu(x)

        x = self.FCEnd(x)

        return x

class MLPL(nn.Module):
    def __init__(self):
        super().__init__()
        self.FCBegin = nn.Linear(2, 10)
        self.FC2 = nn.Linear(10, 15)
        self.FC3 = nn.Linear(15, 20)
        self.FC4 = nn.Linear(20, 10)
        self.FCEnd = nn.Linear(10, 4)

    def forward(self, x):
        x = self.FCBegin(x)
        x = F.relu(x)

        x = self.FC2(x)
        x = F.relu(x)

        x = self.FC3(x)
        x = F.relu(x)

        x = self.FC4(x)
        x = F.relu(x)

        x = self.FCEnd(x)

        return x

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.FCBegin = nn.Linear(2, 20)
        self.FC2 = nn.Linear(20, 30)
        self.FC3 = nn.Linear(30, 40)
        self.FC4 = nn.Linear(40, 40)
        self.FC5 = nn.Linear(40, 20)
        self.FC6 = nn.Linear(20, 10)
        self.FCEnd = nn.Linear(10, 4)

    def forward(self, x):
        x = self.FCBegin(x)
        x = F.relu(x)

        x = self.FC2(x)
        x = F.relu(x)

        x = self.FC3(x)
        x = F.relu(x)

        x = self.FC4(x)
        x = F.relu(x)

        x = self.FC5(x)
        x = F.relu(x)

        x = self.FC6(x)
        x = F.relu(x)

        x = self.FCEnd(x)

        return x

class MyDataset(Dataset):
    def __init__(self, data, label):
        super().__init__()
        self._data = data
        self._label = label

    def __getitem__(self, idx):
        data = self._data[idx]
        label = self._label[idx]
        target = [data, label]
        return target

    def __len__(self):
        return len(self._data)

def load_dataset(source_data, batch_size):
    data = source_data[["x", "y"]]
    label = source_data[["color"]]

    split_rate = 0.9

    train_data_idx = sample(range(len(data)), round(split_rate * len(data)))
    test_data_idx = [x for x in range(len(data)) if x not in train_data_idx]

    train_dataset = MyDataset(data.loc[train_data_idx].values, label.loc[train_data_idx].values)
    test_dataset = MyDataset(data.loc[test_data_idx].values, label.loc[test_data_idx].values)

    return DataLoader(train_dataset, batch_size=batch_size), DataLoader(test_dataset, batch_size=batch_size)


def main():
    source_data = pd.read_csv("set.csv")
    batch_cnt = 3
    #epoch_cnt = 100
    learning_rate = 1e-3
    trainDataLoader, testDataLoader = load_dataset(source_data, batch_size = batch_cnt)
    model = MLP()
    lossFunction = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    #train
    lastCorrect = 0.0
    epoch_cnt = 0
    startTime = time.time()
    #for epoch in range(epoch_cnt):
    while lastCorrect <= 0.95 and epoch_cnt <= 1500:
        epoch_cnt += 1
        sum_loss = 0.0
        correct = 0.0
        for data in trainDataLoader:
            inputs, labels = data
            inputs = inputs.float()
            labels = labels.long().squeeze()
            outputs = model(inputs)
            optimizer.zero_grad()
            loss = lossFunction(outputs, labels)
            loss.backward()
            optimizer.step()
            _, id = torch.max(outputs.data, 1)
            sum_loss += loss.data
            correct += torch.sum(id == labels.data)
        if epoch_cnt % 25 == 0:
            print('epoch: %d loss:%.03f' % (epoch_cnt, sum_loss / len(trainDataLoader)))
            print('          correct:%.03f%%' % (100 * correct / (len(trainDataLoader) * batch_cnt)))
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
        lastCorrect = correct / (len(trainDataLoader) * batch_cnt)
    endTime = time.time()
    print("     total time: %s s" % (endTime - startTime))
    print("    total epoch: %d" % (epoch_cnt))
    print("each epoch cost: %f s/epoch" % ((endTime - startTime) / epoch_cnt))

    #test
    model.eval()
    test_correct = 0.0
    for data in testDataLoader:
        inputs, labels = data
        inputs = inputs.float()
        labels = labels.long().squeeze()
        outputs = model(inputs)
        _, id = torch.max(outputs.data, 1)
        test_correct += torch.sum(id == labels.data)
    print("        correct: %.03f%%" % (100 * test_correct / (len(testDataLoader) * batch_cnt)))

    #visualize
    mat = np.zeros((300, 300))
    for y in range(300):
        for x in range(300):
            inputs = torch.tensor([x, y]).float()
            outputs = model(inputs)
            _, id = torch.max(outputs, 0)
            mat[y, x] = id
    PTcolors = [(255 / 255, 24 / 255, 0 / 255, 0.7),
                (255 / 255, 156 / 255, 0 / 255, 0.7),
                (0 / 255, 193 / 255, 43 / 255, 0.7),
                (14 / 255, 83 / 255, 167 / 255, 0.7)]
    CMcolors = [(255 / 255, 24 / 255, 0 / 255, 0.5),
              (255 / 255, 156 / 255, 0 / 255, 0.5),
              (0 / 255, 193 / 255, 43 / 255, 0.5),
              (14 / 255, 83 / 255, 167 / 255, 0.5)]
    colorMap = LinearSegmentedColormap.from_list("4Dot", CMcolors)
    for i in range(len(source_data)):
        plt.scatter(source_data.loc[i, "x"], source_data.loc[i, "y"], color=PTcolors[source_data.loc[i, "color"]], s=50)
    mat[299, 299] = 3
    mat[0, 0] = 0
    plt.imshow(mat, cmap=colorMap)
    plt.colorbar()
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()


