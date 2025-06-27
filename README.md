# Easy Layers
Make Models Faster with solely torch-based modules

## Installation

### From PyPI (once published)
```bash
pip install easy_layers
```

### From source
```bash
git clone https://github.com/yourusername/easy_layers.git
cd easy_layers
pip install -e .
```

## Usage

```python
from easy_layers.layers import Layer

# Create a basic layer
layer = Layer(name="my_layer")
output = layer(input_data)
```

## Development

### Build the package
```bash
pip install build
python -m build
```

### Install in development mode
```bash
pip install -e .
```
