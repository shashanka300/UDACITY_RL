import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingQNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed, hidden_layers = [64,64]):
        # parameters and  model.
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, hidden_layers[0])
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.relu2 = nn.ReLU()
        

        self.action_size= action_size
        
        self.value_function_fc = nn.Linear(hidden_layers[1], 1)
        
        self.advantage_function_fc = nn.Linear(fc2_units, action_size)
        
    def forward(self, state):
        """Build a network that maps state -> action values."""
        state = self.fc1(state)
       
        state = self.fc2(state)
        
        value_function = self.value_function_fc(state)
        advantage_function = self.advantage_function_fc(state)
        
        return value_function + advantage_function - advantage_function.mean(1).unsqueeze(1).expand(x.size(0), self.action_size) / self.action_size
     

