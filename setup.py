"""
Setup script for PiVR swap correction pipeline.
"""

from setuptools import setup, find_packages

setup(
    name="swap_corrector",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "pytest>=6.2.0"
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A package for correcting head-tail swaps in animal tracking data",
    keywords="animal tracking, data correction, movement analysis",
    python_requires=">=3.7"
) 