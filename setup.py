from setuptools import setup, find_packages
import re

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

long_description = re.sub(r"!\[.*?\]\(.*?\)\n", "", long_description)

setup(
    name="lstm_forecast",
    version="0.1.3",
    author="Gianluca Mazza",
    author_email="gmazza1989@proton.me",
    description="A package for LSTM-based financial time series forecasting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gianlucamazza/lstm_forecast",
    project_urls={
        "Bug Tracker": "https://github.com/gianlucamazza/lstm_forecast/issues",
        "Documentation": "https://github.com/gianlucamazza/lstm_forecast#readme",
        "Source Code": "https://github.com/gianlucamazza/lstm_forecast",
    },
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.7",
    install_requires=[
        "pandas>=2.2.2",
        "ta>=0.11.0",
        "statsmodels>=0.14.2",
        "numpy>=1.26.4",
        "yfinance>=0.2.41",
        "matplotlib>=3.9.1",
        "torch>=2.5.0",
        "plotly>=5.3.1",
        "scikit-learn>=1.5.1",
        "xgboost>=2.1.0",
        "optuna>=3.6.1",
        "onnxruntime>=1.18.1",
        "onnx>=1.16.2",
        "Flask>=3.0.3",
    ],
    extras_require={
        "dev": [
            "pytest>=8.3.2",
            "sphinx>=7.4.7",
            "twine>=5.1.1",
            "black>=24.8.0",
            "flake8>=7.1.1",
            "pre-commit>=3.7.1",
        ],
    },
    entry_points={
        "console_scripts": [
            "lstm_forecast=lstm_forecast.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.html", "*.png"],
    },
    extras_require={
        "dev": [
            "pytest",
            "sphinx",
            "twine",
        ],
    },
    keywords="lstm forecasting finance time series deep learning",
)
