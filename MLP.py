import torch.nn as nn

# Define model hyperparameters
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim):
        """
        Initializes MLP with layers.

        Args:
            input_dim (int): The dimensions of the input
            hidden_dim (int): The dimensions of the hidden layer 
            hidden_dim1 (int): The dimensions of the hidden layer 1
            hidden_dim2 (int): The dimensions of the hidden layer 2
            hidden_dim3 (int): The dimensions of the hidden layer 3
            output_dim (int): The dimensions of the ouput
        """
         
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim1)
        self.relu2 = nn.SiLU()
        self.layernorm = nn.LayerNorm(hidden_dim1)
        self.fc3 = nn.Linear(hidden_dim1, hidden_dim2)
        self.dropout = nn.Dropout(0.2)
        self.relu3 = nn.SiLU()
        self.fc4 = nn.Linear(hidden_dim2, hidden_dim3)
        self.relu4 = nn.SiLU()
        self.fc5 = nn.Linear(hidden_dim3, output_dim)
        self.dropout2 = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Performs a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor
        """
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.layernorm(out)
        out = self.fc3(out)
        out = self.dropout(out)
        out = self.relu3(out)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc5(out)
        out = self.dropout2(out)
        out = self.sigmoid(out)
        
        return out