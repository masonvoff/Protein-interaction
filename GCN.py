import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from torch.utils.data import DataLoader
from GC import GraphConvolution

# Create a graph based on input data from HI-union.tsv
G = nx.read_edgelist('HI-union.tsv')
adj = torch.Tensor(nx.adjacency_matrix(G).todense())

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
class GraphData(torch.utils.data.Dataset):
    def __init__(self, adj):
        self.adj = adj
        
    def __getitem__(self, index):
        return self.adj
    
    def __len__(self):
        return 1
    
data_loader = DataLoader(GraphData(adj), batch_size=1, shuffle=True)

# Loss Function and Optimizer and accuracy function
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(GCN.parameters(), lr=0.1)

def accuracy_func(output, labels):
    correct = torch.eq(output, labels).sum().item()
    acc = correct / len(labels)
    return acc
    
    
# Set up Training ( basic for now variable and argument names will need to be changed)
def training(model: torch.nn.Module,
          data_loader: torch.utils.data.DataLoader,
          loss_func: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          accuracy_func,
          device: torch.device):
    
    # Put in training mode
    labels = torch.Tensor([0, 1]) # Needs to be changed a bit I think 0 and 1 work to show either interaction or not
    model.train()
    
    for batch in data_loader:
        batch.to(device)
    
        # Forward pass
        output = model(batch)
    
        # Loss and metrics
        loss = loss_func()
        train_loss += loss.item()
        train_acc += accuracy_func(output, labels)
    
        # Rest of loop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # Calculate average loss and accuracy
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    
    return train_loss, train_acc
    
# Set up Testing ( basic for now variable and argument names will need to be changed)

def testing(model: torch.nn.Module,
          data_loader: torch.utils.data.DataLoader,
          loss_func: torch.nn.Module,
          accuracy_func,
          device: torch.device):
    
    labels = torch.Tensor([0, 1])
    test_loss = 0
    test_acc = 0
    
    model.eval()
    with torch.inference_mode():
        for batch in data_loader:
            batch.to(device)
            labels.to(device)   
        
            output = model(batch)
        
            loss = loss_func(output, labels)
            test_loss += loss.item()
            test_acc += accuracy_func(output, labels)
        
    test_loss /= len(data_loader)
    test_acc /= len(data_loader)
    
    return test_loss, test_acc
        
    