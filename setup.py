#!/usr/bin/env python3
"""
Setup script for X-35 Parsons Airship Hull Optimization Package
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="x35-parsons",
    version="1.0.0",
    author="Galaxy Unmanned Systems LLC",
    author_email="info@galaxyunmanned.com",
    description="Airship Hull Shape Optimization using Parsons' Method",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/jwhite-GUS/X-35_Parsons",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Aerospace",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "viz": [
            "matplotlib>=3.3",
        ],
    },
    entry_points={
        "console_scripts": [
            "x35-optimize=bin.run_opt:main",
            "x35-plot=bin.plot_results:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="airship hull optimization aerodynamics parsons method",
    project_urls={
        "Bug Reports": "https://github.com/jwhite-GUS/X-35_Parsons/issues",
        "Source": "https://github.com/jwhite-GUS/X-35_Parsons",
        "Documentation": "https://github.com/jwhite-GUS/X-35_Parsons#readme",
    },
)
