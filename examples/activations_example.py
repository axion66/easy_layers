#!/usr/bin/env python3
"""
Activations Example - Demonstrating the various activation functions

This example shows how to use the different activation functions provided by easy_layers:
1. GELU - Gaussian Error Linear Unit
2. SiLU - Sigmoid Linear Unit (also known as Swish)
3. ReLU - Rectified Linear Unit
4. GEGLU - Gated GELU
5. SWIGLU - Gated SiLU (SwiGLU)

The Activation class provides a unified interface to all these activation functions.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from easy_layers.nn.activations.acts import Activation, GELU, SiLU, ReLU, GEGLU, SWIGLU


def plot_activation(name, activation_fn, x_range=(-5, 5), num_points=1000):
    """Plot an activation function over a given range."""
    x = torch.linspace(x_range[0], x_range[1], num_points)
    
    # For gated activations, we need to duplicate the input
    if name in ["GEGLU", "SWIGLU"]:
        # Duplicate x for gated activations
        x_gated = torch.stack([x, x], dim=-1).reshape(-1, 2)
        y = activation_fn(x_gated)
    else:
        y = activation_fn(x)
    
    plt.plot(x.numpy(), y.detach().numpy(), label=name)


def main():
    # Create a range of input values
    x = torch.linspace(-5, 5, 1000)
    
    # Initialize plot
    plt.figure(figsize=(12, 8))
    
    # Example 1: Using individual activation classes
    print("Example 1: Using individual activation classes")
    
    # Create activation instances
    gelu = GELU()
    silu = SiLU()
    relu = ReLU()
    
    # Plot activations
    plot_activation("GELU", gelu, x_range=(-5, 5))
    plot_activation("SiLU", silu, x_range=(-5, 5))
    plot_activation("ReLU", relu, x_range=(-5, 5))
    
    # Example 2: Using the Activation class
    print("Example 2: Using the Activation class")
    
    # Create activations using the unified interface
    gelu_unified = Activation("gelu")
    silu_unified = Activation("silu")
    relu_unified = Activation("relu")
    
    # Verify they produce the same outputs
    x_sample = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    print("\nGELU outputs:")
    print("Direct:", gelu(x_sample))
    print("Unified:", gelu_unified(x_sample))
    
    print("\nSiLU outputs:")
    print("Direct:", silu(x_sample))
    print("Unified:", silu_unified(x_sample))
    
    print("\nReLU outputs:")
    print("Direct:", relu(x_sample))
    print("Unified:", relu_unified(x_sample))
    
    # Example 3: Gated activations (GEGLU and SWIGLU)
    print("\nExample 3: Gated activations (GEGLU and SWIGLU)")
    
    # Create gated activation instances
    geglu = GEGLU()
    swiglu = SWIGLU()
    
    # For gated activations, we need to duplicate the input
    # since they split the input into two parts
    x_gated = torch.cat([x_sample.unsqueeze(1), x_sample.unsqueeze(1)], dim=1)
    
    print("\nInput for gated activations:", x_gated)
    print("GEGLU output:", geglu(x_gated))
    print("SWIGLU output:", swiglu(x_gated))
    
    # Create gated activations using the unified interface
    geglu_unified = Activation("geglu")
    swiglu_unified = Activation("swiglu")
    
    print("\nGEGLU unified output:", geglu_unified(x_gated))
    print("SWIGLU unified output:", swiglu_unified(x_gated))
    
    # Plot gated activations
    plot_activation("GEGLU", geglu, x_range=(-5, 5))
    plot_activation("SWIGLU", swiglu, x_range=(-5, 5))
    
    # Finalize plot
    plt.grid(True)
    plt.legend()
    plt.title("Activation Functions")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.savefig("activations.png")
    print("\nActivation functions plot saved as 'activations.png'")


if __name__ == "__main__":
    main() 