import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from GC import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
    
# Set up a dataloader here

    
# Loss Function and Optimizer and accuracy function
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(GCN.parameters(), lr=0.1)

def accuracy_func(output, labels):
    correct = torch.eq(output, labels).sum().item()
    acc = correct / len(labels)
    return acc
    
    
# Set up Training ( basic for now variable and argument names will need to be changed)
def training(model: torch.nn.Module,
          loss_func: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          accuracy_func,
          device: torch.device):
    model.train()
    
    # Put data on target device
    
    # Forward pass
    
    # Loss and metrics
    loss = loss_func()
    train_loss += loss
    train_acc += accuracy_func()
    
    # Rest of loop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
# Set up Testing ( basic for now variable and argument names will need to be changed)

def testing(model: torch.nn.Module,
          loss_func: torch.nn.Module,
          accuracy_func,
          device: torch.device):
    model.eval()
    with torch.inference_mode():
        # Put data on target device
    
        # Forward pass
    
        # Loss and metrics
        loss = loss_func()
        test_loss += loss
        test_acc += accuracy_func()