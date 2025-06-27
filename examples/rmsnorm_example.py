#!/usr/bin/env python3
"""
RMSNorm Example - Demonstrating the Root Mean Square Normalization

This example shows how to use the RMSNorm module from easy_layers:
1. Creating an RMSNorm layer
2. Applying normalization to inputs
3. Visualizing the normalization effect
4. Comparing with other normalization techniques
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from easy_layers.nn.norms.rms import RMSNorm


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Example 1: Basic usage of RMSNorm
    print("Example 1: Basic usage of RMSNorm")
    
    # Create sample input: [batch_size, heads, seq_len, dim]
    batch_size = 2
    heads = 4
    seq_len = 10
    dim = 16
    
    # Create random input
    x = torch.randn(batch_size, heads, seq_len, dim)
    print(f"Input shape: {x.shape}")
    
    # Create RMSNorm layer
    rms_norm = RMSNorm(heads=heads, dim=dim)
    print(f"RMSNorm parameters: heads={heads}, dim={dim}")
    
    # Apply normalization
    normalized = rms_norm(x)
    print(f"Normalized output shape: {normalized.shape}")
    
    # Example 2: Analyzing normalization effect
    print("\nExample 2: Analyzing normalization effect")
    
    # Create a simpler input for analysis
    simple_x = torch.randn(1, heads, 1, dim)
    print(f"Simple input shape: {simple_x.shape}")
    
    # Print statistics before normalization
    print("Before normalization:")
    print(f"  Mean: {simple_x.mean(dim=-1)}")
    print(f"  Std: {simple_x.std(dim=-1)}")
    
    # Apply normalization
    simple_normalized = rms_norm(simple_x)
    
    # Print statistics after normalization
    print("After normalization:")
    print(f"  Mean: {simple_normalized.mean(dim=-1)}")
    print(f"  Std: {simple_normalized.std(dim=-1)}")
    
    # Example 3: Visualizing the normalization
    print("\nExample 3: Visualizing the normalization")
    
    # Create a 1D sequence for visualization
    seq_len_viz = 100
    viz_x = torch.sin(torch.linspace(0, 4*np.pi, seq_len_viz)).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    viz_x = torch.cat([viz_x, 2*viz_x, -0.5*viz_x], dim=-1)  # Create 3 features
    print(f"Visualization input shape: {viz_x.shape}")
    
    # Create RMSNorm for visualization
    viz_norm = RMSNorm(heads=1, dim=3)
    
    # Apply normalization
    viz_normalized = viz_norm(viz_x)
    
    # Plot original vs normalized
    plt.figure(figsize=(12, 8))
    
    # Original data
    plt.subplot(2, 1, 1)
    for i in range(3):
        plt.plot(viz_x[0, 0, :, i].numpy(), label=f'Feature {i+1}')
    plt.title("Original Data")
    plt.legend()
    plt.grid(True)
    
    # Normalized data
    plt.subplot(2, 1, 2)
    for i in range(3):
        plt.plot(viz_normalized[0, 0, :, i].detach().numpy(), label=f'Normalized Feature {i+1}')
    plt.title("RMS Normalized Data")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("rmsnorm_visualization.png")
    print("\nRMSNorm visualization saved as 'rmsnorm_visualization.png'")
    
    # Example 4: Comparing with other normalization techniques
    print("\nExample 4: Comparing with other normalization techniques")
    
    # Create input for comparison
    comp_x = torch.randn(batch_size, seq_len, dim)
    print(f"Comparison input shape: {comp_x.shape}")
    
    # Apply different normalizations
    # 1. RMSNorm (adapted for this shape)
    comp_rms = RMSNorm(heads=1, dim=dim)
    comp_rms_out = comp_rms(comp_x.unsqueeze(1)).squeeze(1)
    
    # 2. LayerNorm
    layer_norm = torch.nn.LayerNorm(dim)
    comp_ln_out = layer_norm(comp_x)
    
    # 3. BatchNorm
    batch_norm = torch.nn.BatchNorm1d(dim)
    comp_bn_out = batch_norm(comp_x.transpose(1, 2)).transpose(1, 2)
    
    # Compute statistics
    print("Normalization comparison:")
    print(f"  Original - Mean: {comp_x.mean().item():.4f}, Std: {comp_x.std().item():.4f}")
    print(f"  RMSNorm - Mean: {comp_rms_out.mean().item():.4f}, Std: {comp_rms_out.std().item():.4f}")
    print(f"  LayerNorm - Mean: {comp_ln_out.mean().item():.4f}, Std: {comp_ln_out.std().item():.4f}")
    print(f"  BatchNorm - Mean: {comp_bn_out.mean().item():.4f}, Std: {comp_bn_out.std().item():.4f}")


if __name__ == "__main__":
    main() 