from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'A data science roadmap'
LONG_DESCRIPTION = 'A long term project for archiving past handcrafted code for future reference'

setup(
    name="pandora",
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author="Eric Pfleiderer",
    author_email="e.pfleiderer@hotmail.com",
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'torch',
        'torchvision'
    ],
    keywords='roadmap',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        'License :: OSI Approved :: MIT License',
        "Programming Language :: Python :: 3",
    ]
)