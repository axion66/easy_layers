#!/usr/bin/env python3
"""
FeedForward Example - Demonstrating the FeedForward layer

This example shows how to use the FeedForward layer from easy_layers:
1. Creating a FeedForward layer with different activation functions
2. Processing input through the layer
3. Visualizing the transformation
4. Comparing different configurations
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from easy_layers.nn.layers.layer import FeedForward


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create sample input
    batch_size = 8
    in_features = 16
    x = torch.randn(batch_size, in_features)
    
    print(f"Input shape: {x.shape}")
    
    # Example 1: Basic FeedForward layer
    print("\nExample 1: Basic FeedForward layer")
    
    # Create a FeedForward layer with default parameters
    ff_default = FeedForward(
        in_features=in_features,
        activation="geglu"  # Default activation
    )
    
    # Print layer information
    print(f"Layer: {ff_default}")
    print(f"Total parameters: {sum(p.numel() for p in ff_default.parameters())}")
    
    # Process input
    output_default = ff_default(x)
    print(f"Output shape: {output_default.shape}")
    
    # Example 2: Custom FeedForward layer
    print("\nExample 2: Custom FeedForward layer")
    
    # Create a FeedForward layer with custom parameters
    ff_custom = FeedForward(
        in_features=in_features,
        out_features=8,  # Different output dimension
        hidden_features=32,  # Custom hidden dimension
        dropout=0.1,  # Lower dropout
        activation="swiglu"  # Different activation
    )
    
    # Print layer information
    print(f"Layer: FeedForward(in_features={in_features}, out_features=8, hidden_features=32, dropout=0.1, activation=swiglu)")
    print(f"Total parameters: {sum(p.numel() for p in ff_custom.parameters())}")
    
    # Process input
    output_custom = ff_custom(x)
    print(f"Output shape: {output_custom.shape}")
    
    # Example 3: Comparing different activation functions
    print("\nExample 3: Comparing different activation functions")
    
    # Create FeedForward layers with different activations
    activations = ["gelu", "silu", "relu", "geglu", "swiglu"]
    ff_layers = {
        act: FeedForward(
            in_features=in_features,
            out_features=in_features,
            activation=act,
            dropout=0.0  # No dropout for comparison
        ) for act in activations
    }
    
    # Process the same input through all layers
    outputs = {act: layer(x) for act, layer in ff_layers.items()}
    
    # Compute statistics for comparison
    stats = {}
    for act, output in outputs.items():
        stats[act] = {
            "mean": output.mean().item(),
            "std": output.std().item(),
            "min": output.min().item(),
            "max": output.max().item()
        }
    
    # Print statistics
    for act, stat in stats.items():
        print(f"{act.upper()}: mean={stat['mean']:.4f}, std={stat['std']:.4f}, min={stat['min']:.4f}, max={stat['max']:.4f}")
    
    # Example 4: Visualizing the transformation
    print("\nExample 4: Visualizing the transformation")
    
    # Create a 1D input for visualization
    x_1d = torch.linspace(-3, 3, 100).unsqueeze(1)  # Shape: [100, 1]
    
    # Create a simple FeedForward for 1D input
    ff_viz = FeedForward(
        in_features=1,
        out_features=1,
        hidden_features=16,
        dropout=0.0,  # No dropout for visualization
        activation="gelu"
    )
    
    # Process input
    output_1d = ff_viz(x_1d)
    
    # Plot input vs output
    plt.figure(figsize=(10, 6))
    plt.plot(x_1d.detach().numpy(), x_1d.detach().numpy(), 'r--', label='Input (identity)')
    plt.plot(x_1d.detach().numpy(), output_1d.detach().numpy(), 'b-', label='FeedForward output')
    plt.grid(True)
    plt.legend()
    plt.title("FeedForward Transformation")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.savefig("feedforward_transform.png")
    print("\nFeedForward transformation plot saved as 'feedforward_transform.png'")


if __name__ == "__main__":
    main() 