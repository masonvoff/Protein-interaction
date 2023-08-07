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

    list1 = []
    list2 = []

    for line in lines:
        columns = line.strip().split('\t')
        if len(columns) >= 2:
            list1.append(columns[0][4:])
            list2.append(columns[1][4:])


    # Create dictionaries
    unique_elements = sorted(list(set(list1 + list2))) # Needs to be sorted to ensure same order
    category_to_index = {category: index for index, category in enumerate(unique_elements)}
    index_to_category = {index: category for category, index in category_to_index.items()}

    # Convert strings to values
    numeric_list1 = [category_to_index[val] for val in list1]
    numeric_list2 = [category_to_index[val] for val in list2]
    
    # Convert the numeric lists to numpy arrays
    array1 = np.array(numeric_list1, dtype=np.int64)
    array2 = np.array(numeric_list2, dtype=np.int64)
    
    # Convert dtype to float32 while preserving the integer values
    array1 = array1.astype(np.float32)
    array2 = array2.astype(np.float32)

    # Convert the numpy arrays to PyTorch tensors ( should presereve the float32 dtype )
    tensor1 = torch.from_numpy(array1)
    tensor2 = torch.from_numpy(array2)

    print(tensor1)
    print(tensor2)
    

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    readDataNx2(FILENAME)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
