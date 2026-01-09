"""Setup configuration for quant-modelling package."""

from setuptools import setup, find_packages

setup(
    name='quant-modelling',
    version='0.1.0',
    description='Quantitative modeling library for financial derivatives pricing',
    author='QuantAnalyticsTorch',
    packages=find_packages(exclude=['tests', 'tests.*']),
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.20.0',
        'scipy>=1.7.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Financial and Insurance Industry',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
