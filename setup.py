# setup.py for initializing a lib

from setuptools import setup, find_packages

setup(
    name='cropcircles',
    version='0.1',
    packages=find_packages(),
    python_requires='>=3.10',
    install_requires=[
        'numpy',
        'pandas',
        'torch',
        'tensorflow',
        'scikit-learn',
        'torchinfo',
        'horovod'
    ],
    description='Deep Learning library for tabular data problems',
    author='Reisen Raumberg',
    author_email='fallturm.bremen@gmail.com',
    url='https://github.com/Raumberg/crop-circles',
)