# setup.py

from setuptools import setup, find_packages

setup(
    name='DadilipTrainer',
    version='0.1.0',
    description='A computer vision model trainer package.',
    author='Dadilip Inc.',
    author_email='dadilip@gmail.com',
    url='https://github.com/drapraks/DadilipTrainer',  # Replace with your repository URL
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'tensorflow',
        'matplotlib',
        'scikit-learn',
        'Pillow'
    ],
    python_requires='>=3.6',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
