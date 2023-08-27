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
import networkx as nx
import matplotlib.pyplot as plt
#import tqdm

FILENAME = "HI-union.tsv"


def readDataNx2(name):
    # reads data from file into tensors
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
    np_array1 = np.array(numeric_list1, dtype=np.float32)
    np_array2 = np.array(numeric_list2, dtype=np.float32)

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
    def __init__(self, input_size, hidden_size, output_size):
        super(GCNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, adjacency_matrix, features):
        x = torch.mm(adjacency_matrix, features)
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

def graphData(tensor1,tensor2):
    G = nx.DiGraph()

    # Add nodes

    G.add_node(tensor1[0].item())

    # Add edges

    neighbors = graph.get_neighbors(tensor1[0].item())
    for neighbor in neighbors:
        G.add_edge(tensor1[0].item(), neighbor.item())

    # Draw the graph
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=1000, node_color="skyblue", font_size=10, font_color="black",
            font_weight="bold")
    plt.title("Graph Visualization")
    plt.show()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, Y = readDataNx2(FILENAME)
    print(torch.__version__)
    graph = Graph(X, Y)
    # Get neighbors a tensor1
    tensor1_index = 457.0
    neighbors = graph.get_neighbors(tensor1_index)
    print(neighbors)
    tensor1 = X
    tensor2 = Y
    graph = Graph(tensor1, tensor2)
    graphData(tensor1,tensor2)



    num_tensor1 = len(tensor1)
    num_tensor2 = len(tensor2)
    adjacency_matrix = torch.zeros((num_tensor1, num_tensor2), dtype=torch.float)

    for i in range(num_tensor1):
        neighbors = graph.get_neighbors(tensor1[i].item())
        for neighbor in neighbors:
            j = (tensor2 == neighbor).nonzero(as_tuple=False).squeeze()
            adjacency_matrix[i, j] = 1

    # Hyperparameters
    input_size = 1
    hidden_size = 800
    output_size = 1
    learning_rate = 1
    num_epochs = 100

    # Create the GCN model
    model = GCNet(input_size, hidden_size, output_size)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Prepare features tensor
    features = tensor2.view(-1, 1).float()

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





    #accuracy = lambda y_pred, y_true: torch.mean((torch.argmax(y_pred, dim=1) == y_true).float())