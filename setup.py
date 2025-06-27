from setuptools import setup, find_packages
import os
import re

# Read the version from __init__.py
with open(os.path.join("easy_layers", "__init__.py"), encoding="utf-8") as f:
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if version_match:
        version = version_match.group(1)
    else:
        raise RuntimeError("Unable to find version string.")

# Read the README for the long description
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="easy_layers",
    version=version,
    author="Your Name",
    author_email="your.email@example.com",
    description="A simple neural network layers library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/easy_layers",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        # Add your dependencies here
    ],
) 