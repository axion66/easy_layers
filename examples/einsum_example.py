#!/usr/bin/env python3
"""
UltEinsum Example - Demonstrating the UltEinsum module for tensor operations

This example shows how to use the UltEinsum module for:
1. Reshaping tensors using einops patterns
2. Performing tensor multiplication using einsum notation
3. Using UltEinsum in a neural network
4. Complex reshaping operations with einops

UltEinsum provides a unified interface for both reshaping operations (using einops.rearrange)
and tensor multiplication (using torch.einsum).
"""

import torch
from easy_layers.nn.einsum import UltEinsum


def main():
    # Create some example tensors
    # 2x3 matrix
    tensor_a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    print("Tensor A (2x3):")
    print(tensor_a)
    
    # 3x2 matrix
    tensor_b = torch.tensor([[7, 8], [9, 10], [11, 12]], dtype=torch.float32)
    print("\nTensor B (3x2):")
    print(tensor_b)
    
    # Example 1: Reshape operation (transpose a 2x3 matrix to 3x2)
    print("\n--- Reshape Example using einops.rearrange ---")
    reshape_op = UltEinsum("b c -> c b", mode='r')  # einops pattern
    transposed = reshape_op(tensor_a)
    print("Transposed A (3x2):")
    print(transposed)
    
    # Example 2: Matrix multiplication (2x3 @ 3x2 = 2x2)
    print("\n--- Multiplication Example using einsum ---")
    matmul_op = UltEinsum("ik,kj->ij", mode='m')
    result = matmul_op(tensor_a, tensor_b)
    print("A @ B (2x2):")
    print(result)
    
    # Compare with standard matrix multiplication
    print("\nStandard torch.matmul result:")
    print(torch.matmul(tensor_a, tensor_b))
    
    # Example 3: Using static methods
    print("\n--- Using Static Methods ---")
    
    # Reshape using static method with einops pattern
    transposed_static = UltEinsum.reshape("h w -> w h", tensor_a)
    print("Transposed A using static method:")
    print(transposed_static)
    
    # Multiply using static method
    result_static = UltEinsum.multiply("ik,kj->ij", tensor_a, tensor_b)
    print("A @ B using static method:")
    print(result_static)
    
    # Example 4: Batch matrix multiplication
    print("\n--- Batch Matrix Multiplication ---")
    # Create batch tensors (2x2x3 and 2x3x2)
    batch_a = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=torch.float32)
    batch_b = torch.tensor([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]], dtype=torch.float32)
    
    print("Batch A shape:", batch_a.shape)
    print("Batch B shape:", batch_b.shape)
    
    # Batch matrix multiplication
    batch_matmul = UltEinsum("bik,bkj->bij", mode='m')
    batch_result = batch_matmul(batch_a, batch_b)
    print("Batch result shape:", batch_result.shape)
    print("Batch result:")
    print(batch_result)
    
    # Example 5: Using UltEinsum in a Neural Network
    print("\n--- Using UltEinsum in a Neural Network ---")
    
    class SimpleNetwork(torch.nn.Module):
        def __init__(self):
            super(SimpleNetwork, self).__init__()
            self.einsum_op = UltEinsum("ik,kj->ij", mode='m')
            
        def forward(self, x, weight):
            return self.einsum_op(x, weight)
    
    # Create a simple network
    net = SimpleNetwork()
    
    # Use the network
    output = net(tensor_a, tensor_b)
    print("Network output:")
    print(output)
    
    # Example 6: More complex reshaping with einops
    print("\n--- Complex Reshaping with einops patterns ---")
    
    # Create a 2x3x4 tensor
    tensor_c = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    print("Tensor C (2x3x4):")
    print(tensor_c)
    
    # Reshape to 6x4
    flatten_op = UltEinsum("b c d -> (b c) d", mode='r')
    flattened = flatten_op(tensor_c)
    print("\nFlattened C (6x4):")
    print(flattened)
    
    # Reshape to 4x6
    # We need to provide the dimensions for b and c
    transpose_op = UltEinsum("(b c) d -> d (b c)", mode='r', b=2, c=3)
    transposed_c = transpose_op(flattened)
    print("\nTransposed C (4x6):")
    print(transposed_c)
    
    # Reshape back to original shape
    reshape_back_op = UltEinsum("d (b c) -> b c d", mode='r', b=2, c=3)
    reshaped_back = reshape_back_op(transposed_c)
    print("\nReshaped back to 2x3x4:")
    print(reshaped_back)


if __name__ == "__main__":
    main() 