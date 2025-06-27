"""
UltEinsum - A versatile implementation that supports
both reshaping operations (using einops.rearrange) and tensor multiplication (using einsum).
"""

import torch
import torch.nn as nn
from einops import rearrange
from typing import Literal, Optional

def UltEinsum_functional(
    equation: str,
    tensor1: torch.Tensor,
    tensor2: Optional[torch.Tensor] = None,
    mode: Literal["r", "m"] = "m",
    **kwargs,
):
    """
    Functional version of UltEinsum.
    
    Args:
        equation (str): Equation for the operation. For mode 'r', this is an einops pattern.
                        For mode 'm', this is an einsum equation.
        tensor1: First input tensor.
        tensor2: Second input tensor (only used in multiplication mode).
        mode (str): Operation mode, either 'r' for reshape or 'm' for multiplication.
        **kwargs: Additional keyword arguments for einops.rearrange when mode='r'.
    
    Returns:
        torch.Tensor: Result of the operation.
    """
    if mode == 'r':
        return rearrange(tensor1, equation, **kwargs)
    elif mode == 'm':
        if tensor2 is None:
            raise ValueError("Multiplication mode requires two tensors")
        return torch.einsum(equation, tensor1, tensor2)
    else:
        raise ValueError("Mode must be either 'r' for reshape or 'm' for multiplication")


class UltEinsum(nn.Module):
    """
    UltEinsum (Ultimate Einstein Summation) module that provides a unified interface
    for both reshaping operations (using einops.rearrange) and tensor multiplication (using einsum).
    
    Modes:
    - 'r': Reshape a single tensor using einops.rearrange
    - 'm': Tensor multiplication between multiple tensors using torch.einsum
    """
    
    def __init__(
        self,
        equation: str,
        mode: Literal["r", "m"] = "m",
        **kwargs,
    ):
        """
        Initialize the UltEinsum operation.
        
        Args:
            equation (str): For mode 'r', this is an einops pattern.
                           For mode 'm', this is an einsum equation.
            mode (str): Operation mode, either 'r' for reshape or 'm' for multiplication.
            **kwargs: Additional keyword arguments for einops.rearrange when mode='r'.
        """
        super(UltEinsum, self).__init__()
        self.equation = equation
        if mode not in ['r', 'm']:
            raise ValueError("Mode must be either 'r' for reshape or 'm' for multiplication")
        self.mode = mode
        self.kwargs = kwargs
        
    def forward(self, *tensors):
        """
        Apply the operation to the input tensors.
        
        Args:
            *tensors: Input tensors to apply the operation to.
            
        Returns:
            torch.Tensor: Result of the operation.
        """
        if self.mode == 'r':
            if len(tensors) != 1:
                raise ValueError("Reshape mode ('r') requires exactly one input tensor")
            return self._reshape(tensors[0])
        else:  # mode == 'm'
            return self._multiply(*tensors)
    
    def _reshape(self, tensor):
        """
        Reshape a single tensor using einops.rearrange.
        
        Args:
            tensor (torch.Tensor): Input tensor to reshape.
            
        Returns:
            torch.Tensor: Reshaped tensor.
        """
        try:
            return rearrange(tensor, self.equation, **self.kwargs)
        except Exception as e:
            print(f"Error in reshape operation: {e}")
            raise
    
    def _multiply(self, *tensors):
        """
        Multiply tensors using einsum notation.
        
        Args:
            *tensors: Input tensors to multiply.
            
        Returns:
            torch.Tensor: Result of the multiplication.
        """
        try:
            return torch.einsum(self.equation, *tensors)
        except Exception as e:
            print(f"Error in multiplication operation: {e}")
            raise
    
    @staticmethod
    def reshape(pattern, tensor, **kwargs):
        """
        Static method for reshaping a tensor using einops.rearrange.
        
        Args:
            pattern (str): Einops rearrange pattern.
            tensor (torch.Tensor): Input tensor to reshape.
            **kwargs: Additional keyword arguments for einops.rearrange.
            
        Returns:
            torch.Tensor: Reshaped tensor.
        """
        module = UltEinsum(pattern, 'r', **kwargs)
        return module(tensor)
    
    @staticmethod
    def multiply(equation, *tensors):
        """
        Static method for multiplying tensors using einsum.
        
        Args:
            equation (str): Einstein summation equation.
            *tensors: Input tensors to multiply.
            
        Returns:
            torch.Tensor: Result of the multiplication.
        """
        module = UltEinsum(equation, 'm')
        return module(*tensors)
