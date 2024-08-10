from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='lstm_forecast',
    version='0.1.1',
    author='Gianluca Mazza',
    author_email='gmazza1989@proton.me',
    description='A package for LSTM-based financial time series forecasting',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/gianlucamazza/lstm_forecast',
    project_urls={
        'Bug Tracker': 'https://github.com/gianlucamazza/lstm_forecast/issues',
        'Documentation': 'https://github.com/gianlucamazza/lstm_forecast#readme',
        'Source Code': 'https://github.com/gianlucamazza/lstm_forecast',
    },
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Operating System :: OS Independent',
    ],
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.7',
    install_requires=[
        'pandas',
        'ta',
        'statsmodels',
        'numpy',
        'yfinance',
        'matplotlib',
        'torch',
        'plotly',
        'scikit-learn',
        'xgboost',
        'optuna',
        'onnxruntime',
        'onnx',
        'flask',
    ],
    entry_points={
        'console_scripts': [
            'lstm_forecast=lstm_forecast.cli:main',
        ],
    },
    include_package_data=True,
    package_data={
        '': ['*.json', '*.html', '*.png'],
    },
    extras_require={
        'dev': [
            'pytest',
            'sphinx',
            'twine',
        ],
    },
    keywords='lstm forecasting finance time series deep learning',
)
