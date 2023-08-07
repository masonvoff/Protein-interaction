import torch
import numpy as np
from torch import nn
from numpy import genfromtxt

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
    np_array1 = np.array(numeric_list1)
    np_array2 = np.array(numeric_list2)
    
    # onvert from np arrays to tensors
    tensor1 = torch.from_numpy(np_array1)
    tensor2 = torch.from_numpy(np_array2)
    
    # Print first 10 elements of tensors
    print(f"Tensor1: {tensor1[:10]}, Tensor2: {tensor2[:10]} ")
    print(f"Tensor DTypes: {tensor1.dtype}, {tensor2.dtype}")
    print(f"Tensor Shapes: {tensor1.shape}, {tensor2.shape}")
    
    
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    readDataNx2(FILENAME)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
