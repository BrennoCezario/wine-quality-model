import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim): # Método construtor da classe SimpleMLP
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x): # Método que define o fluxo de dados na rede
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x