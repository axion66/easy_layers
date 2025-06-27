import torch
import torch.nn as nn
from easy_layers.nn.activations.acts import Activation
from typing import Optional




class FeedForward(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: Optional[int] = None,
        hidden_features: Optional[int] = None,
        dropout: float = 0.2,
        activation: str = "geglu",
        init_range: float = 0.02,
        bias_layer1: bool = True,
        bias_layer2: bool = True,
        default_in_to_hidden_ratio: float = 4,
    ):
        super().__init__()
        self.init_weights(init_range)
        
        if out_features is None:
            out_features = in_features
            
        if hidden_features is None:
            hidden_features = default_in_to_hidden_ratio * in_features
            
        if activation == "geglu" or activation == "swiglu":
            self.hidden_features = hidden_features * 2
        else:
            self.hidden_features = hidden_features
            
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.activation = activation
        self.bias_layer1 = bias_layer1
        self.bias_layer2 = bias_layer2
        self.default_in_to_hidden_ratio = default_in_to_hidden_ratio
        
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features, bias=bias_layer1),
            Activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, out_features, bias=bias_layer2),
        )
       
       
    def init_weights(self, init_range: float = 0.02):
        nn.init.uniform_(self.mlp[0].weight, -init_range, init_range)
        nn.init.uniform_(self.mlp[2].weight, -init_range, init_range)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.zeros_(self.mlp[2].bias)
        
    def forward(self, x):
        return self.mlp(x)
    
    def __repr__(self):
        return f"FeedForward(in_features={self.in_features}, out_features={self.out_features}, hidden_features={self.hidden_features}, dropout={self.dropout}, activation={self.activation}, bias_layer1={self.bias_layer1}, bias_layer2={self.bias_layer2}, default_in_to_hidden_ratio={self.default_in_to_hidden_ratio})"