# https://github.com/tkipf/pygcn/blob/master/pygcn/utils.py can be adapted for our data
import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from collections import defaultdict
import GCNetwork
from GCNetwork import GCN



FILENAME = "HI-union.tsv"

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


class Graph:
    def __init__(self, tensor1, tensor2):
        self.tensor1 = tensor1
        self.tensor2 = tensor2
        self._construct_graph()

    def _construct_graph(self):
        self.graph = {}

        for i, value in enumerate(self.tensor1):
            value = value.item()
            if value not in self.graph:
                self.graph[value] = []
            self.graph[value].append(self.tensor2[i])

    def get_neighbors(self, tensor1_value):
        if tensor1_value in self.graph:
            return self.graph[tensor1_value]
        else:
            return []
class GCNet(nn.Module):
    def __init__(self, num_features, hidden_size, num_classes):
        super(GCNet, self).__init__()
        self.conv1 = nn.Linear(num_features, hidden_size)
        self.conv2 = nn.Linear(hidden_size, num_classes)

    def forward(self, adjacency_matrix, features):
        x = torch.mm(adjacency_matrix, features)
        x = torch.relu(self.conv1(x))
        x = self.conv2(x)
        return x



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, Y = readDataNx2(FILENAME)
    print(torch.__version__)

    graph = Graph(X, Y)

    #Get neighbors a tensor1
    tensor1_index = 457.0
    neighbors = graph.get_neighbors(tensor1_index)
    print(neighbors)

    tensor1 = X
    tensor2 = Y

    # Create the graph instance
    graph = Graph(tensor1, tensor2)

    # Construct adjacency matrix
    num_nodes = len(tensor1)
    adjacency_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float)

    for i in range(num_nodes):
        neighbors = graph.get_neighbors(tensor1[i].item())
        for neighbor in neighbors:
            j = (tensor2 == neighbor).nonzero()
            adjacency_matrix[i, j] = 1

    # Hyperparameters
    input_size = 1
    hidden_size = 16
    output_size = 1
    learning_rate = 0.01
    num_epochs = 100

    # Create the GCN model
    model = GCNet(input_size, hidden_size, output_size)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Prepare features tensor
    features = tensor1.view(-1, 1).float()

    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(adjacency_matrix, features)
        loss = criterion(outputs, features)
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Make predictions
    model.eval()
    with torch.no_grad():
        predicted_features = model(adjacency_matrix, features)
        print("Predicted Features:")
        print(predicted_features)









    myGCN = GCN(64006,10,64006,10)

    # Loss Function and Optimizer and accuracy function
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(myGCN.parameters(), lr=0.1)



