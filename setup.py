"""
Setup script to install project as package
"""

from setuptools import setup, find_packages

setup(
    name="rnn-sentiment-analysis",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
    ],
)