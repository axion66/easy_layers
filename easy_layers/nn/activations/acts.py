import torch
import torch.nn as nn

class Activation(nn.Module):
    def __init__(
        self,
        name: str,
        **kwargs,
    ):
        super().__init__()
        self.name = name
        name = name.lower()
        if name == "gelu":
            self.act = GELU()
        elif name == "silu":
            self.act = SiLU()
        elif name == "relu":
            self.act = ReLU()
        elif name == "geglu":
            self.act = GEGLU()
        elif name == "swiglu":
            self.act = SWIGLU()
        else:
            raise ValueError(f"Activation {name} not found. We have only GELU, SiLU, ReLU, GEGLU, and SWIGLU.")
         
    def forward(self, x):
        return self.act(x)
  
  
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return torch.nn.functional.gelu(x)
    
    
class SiLU(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return torch.nn.functional.silu(x)
    
    
class ReLU(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return torch.nn.functional.relu(x)
    
    
class GEGLU(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return torch.nn.functional.gelu(gate) * x
    
    
class SWIGLU(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * x
    
  
