# https://github.com/tkipf/pygcn/blob/master/pygcn/utils.py can be adapted for our data
import torch
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
        self.adjacency_matrix = self._create_adjacency_matrix()

    def _create_adjacency_matrix(self):
        num_tensor1 = len(self.tensor1)
        num_tensor2 = len(self.tensor2)
        adjacency_matrix = torch.zeros((num_tensor1, num_tensor2), dtype=torch.float)

        for i in range(num_tensor1):
            adjacency_matrix[i, i] = 1

        return adjacency_matrix

    def get_neighbors(self, tensor1_index):
        return self.tensor2[tensor1_index]


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, Y = readDataNx2(FILENAME)
    print(torch.__version__)

    graph = Graph(X, Y)

    # Example: Get neighbors of a tensor1 element
    tensor1_index = 0  # Replace with the index you're interested in
    neighbors = graph.get_neighbors(tensor1_index)
    print(neighbors)









    myGCN = GCN(64006,10,64006,10)

    # Loss Function and Optimizer and accuracy function
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(myGCN.parameters(), lr=0.1)



