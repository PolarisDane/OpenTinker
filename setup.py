#!/usr/bin/env python3
"""
Setup script for OpenTinker.

This allows the package to be installed in development mode with:
    pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read requirements
def read_requirements(filename):
    """Read requirements from file, ignoring comments and empty lines."""
    requirements = []
    filepath = Path(__file__).parent / filename
    if filepath.exists():
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    requirements.append(line)
    return requirements


# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()


setup(
    name="opentinker",
    version="0.1.0",
    description="OpenTinker: A distributedframework for training and inference with interactive environments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="OpenTinker Team",
    python_requires=">=3.8",
    packages=find_packages(include=["opentinker", "opentinker.*"]),
    install_requires=[
        # Core dependencies
        "ray>=2.9.0",
        "torch>=2.0.0",
        "transformers>=4.35.0",
        # Web framework
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "pydantic>=2.0.0",
        # Configuration
        "omegaconf>=2.3.0",
        "hydra-core>=1.3.0",
        "pyyaml>=6.0",
        # Data processing
        "pandas>=2.0.0",
        "pyarrow>=14.0.0",
        "datasets>=2.14.0",
        # Utilities
        "requests>=2.31.0",
        "aiohttp>=3.9.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.1.0",
        ],
        "logging": [
            "wandb>=0.16.0",
        ],
    },
    entry_points={
        # "console_scripts": [
        #     "opentinker-scheduler=opentinker.scheduler.launch_scheduler_kill:main",
        # ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
