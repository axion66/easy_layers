#!/usr/bin/env python3

from easy_layers import __version__
from easy_layers.layers import Layer

def main():
    print(f"Easy Layers version: {__version__}")
    
    # Create a layer
    layer = Layer(name="test_layer")
    
    # Test the layer
    input_data = [1, 2, 3, 4, 5]
    output = layer(input_data)
    
    print(f"Layer name: {layer.name}")
    print(f"Input: {input_data}")
    print(f"Output: {output}")
    print("Layer test successful!")

if __name__ == "__main__":
    main() 