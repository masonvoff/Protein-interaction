import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from numpy import genfromtxt
from sklearn.model_selection import train_test_split



FILENAME = "HI-union.tsv"
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def readDataNx2(name):
    #reads data from file into tensors
    with open(FILENAME, "r") as file:
        lines = file.readlines()
            
    # Initialize empty lists       
    list1 = []
    list2 = []
    
    # Iterate through lines and append to lists
    for line in lines:
        columns = line.strip().split('\t')
        if len(columns) >= 2:
            list1.append(columns[0][4:].lstrip('0'))
            list2.append(columns[1][4:].lstrip('0'))
        
    # Print first 10 elements of lists
    print(f"String List1: {list1[:10]}, String List2: {list2[:10]} ")
    print("\n")
    
    # Convert strings to integer values
    numeric_list1 = [int(val) for val in list1]
    numeric_list2 = [int(val) for val in list2]
    
    # Print first 10 elements of lists
    print(f"Integer List1: {numeric_list1[:10]}, Integer List2: {numeric_list2[:10]} ")
    print("\n")
    
    # Convert lists into np arrays
    np_array1 = np.array(numeric_list1,dtype= np.float32)
    np_array2 = np.array(numeric_list2,dtype= np.float32)
    
    # convert from np arrays to tensors
    tensor1 = torch.from_numpy(np_array1)
    tensor2 = torch.from_numpy(np_array2)
    
    # Print first 10 elements of tensors
    print(f"Tensor1: {tensor1[:10]}, Tensor2: {tensor2[:10]} ")
    print(f"Tensor DTypes: {tensor1.dtype}, {tensor2.dtype}")
    print(f"Tensor Shapes: {tensor1.shape}, {tensor2.shape}")

    return tensor1, tensor2


class BasicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=51204, out_features=5)
        self.layer_2 = nn.Linear(in_features=5, out_features=1)

    def forward(self, x):
        return self.layer_2(self.layer_1(x))

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100
    return acc

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, Y = readDataNx2(FILENAME)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,  random_state=10)  #create test and train data
    print(len(X_train), len(X_test), len(Y_train), len(Y_test))
    Bmodel = BasicModel()
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(params= Bmodel.parameters(), lr=0.1)
    Y_logits = Bmodel(X_test.to(device))
    print(Y_logits)


    epochs = 10000
    for epoch in range(epochs):
        #training loop
        Bmodel.train()
        Y_logits = Bmodel(X_train).squeeze
        Y_pred = torch.round(torch.sigmoid(Y_logits))

        loss = loss_fn(Y_logits,Y_train)
        acc = accuracy_fn(y_true=Y_train, y_pred=Y_pred)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        Bmodel.eval()

        #testing loop
        with torch.inference_mode():
            test_logits = Bmodel(X_test).squeeze()
            test_pred = torch.round(torch.sigmoid(test_logits))
            test_loss = loss_fn(test_logits,Y_test)
            test_acc = accuracy_fn(Y_test, test_pred)

        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")