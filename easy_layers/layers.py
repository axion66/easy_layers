"""Basic neural network layers."""

class Layer:
    """Base class for all layers."""
    
    def __init__(self, name=None):
        """Initialize a layer.
        
        Args:
            name (str, optional): Name of the layer.
        """
        self.name = name
        
    def forward(self, inputs):
        """Forward pass.
        
        Args:
            inputs: Input tensor.
            
        Returns:
            Output tensor.
        """
        return inputs
    
    def __call__(self, inputs):
        """Call the layer."""
        return self.forward(inputs) 