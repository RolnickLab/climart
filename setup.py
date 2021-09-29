from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()
keywords = ["radiative transfer", "emulation", "deep learning", "climart", "dataset", "pytorch", "mila", "eccc"]

setup(
    name='climart',
    version='0.1.0',
    packages=find_packages(),
    url='https://github.com/Venka97/climART',
    license='CC BY 4.0',
    author='Salva Rühling Cachay, Venkatesh Ramesh',
    author_email='',
    keywords=keywords,
    description='A comprehensive, large-scale dataset for'
                ' benchmarking neural network emulators of the'
                ' radiation component in climate and weather models.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.7.0",
    install_requires=[
        "torch>=1.7.1",
        "scikit-learn",
    ],
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
