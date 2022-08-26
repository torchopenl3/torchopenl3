import imp
import os
from setuptools import find_packages, setup

version = imp.load_source(
    "torchopenl3.version", os.path.join("torchopenl3", "version.py")
)

with open("README.md") as file:
    long_description = file.read()

setup(
    name="torchopenl3",
    version=version.version,
    description="Deep audio and image embeddings, based on Look, Listen, and Learn approach Pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/torchopenl3/torchopenl3/",
    author="Humair Raj Khan and Gyanendra Das",
    author_email="gyanendralucky9337@gmail.com",
    packages=find_packages(),
    entry_points={
        "console_scripts": ["torchopenl3=torchopenl3.cli:main"],
    },
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Programming Language :: Python :: 3.6",
    ],
    install_requires=[
        "numpy>=1.13.0",
        "scipy>=0.19.1",
        "soundfile",
        "resampy>=0.2.1,<0.3.0",
        #'scikit-image>=0.14.3,<0.15.0',
        "torch>=1.4.0",
        "nnAudio>=0.2.4",
        "julius>=0.2.5",
        "librosa",
    ],
    extras_require={
        "test": [
            "pytest",
            "pytest-cov",
            "Cython >= 0.23.4",  # For openl3
            # "scikit-image==v0.18.1",    # For openl3
            "openl3==0.3.1",
            "kapre==0.1.4",  # For openl3
            "h5py==2.10.0",  # For openl3
            "tensorflow<1.14",  # For openl3
            "requests",
            "tqdm",
            "protobuf<=3.20.1",  # https://exerror.com/typeerror-descriptors-cannot-not-be-created-directly/
        ],
        "dev": [
            "pytest",
            "pytest-cov",
            "Cython >= 0.23.4",  # For openl3
            # "scikit-image==v0.18.1",    # For openl3
            "openl3==0.3.1",
            "kapre==0.1.4",  # For openl3
            "h5py==2.10.0",  # For openl3
            "tensorflow<1.14",  # For openl3
            "requests",
            "tqdm",
            "pre-commit",
            "nbstripout==0.3.9",  # Used in precommit hooks
            "black==20.8b1",  # Used in precommit hooks
            "jupytext==v1.10.3",  # Used in precommit hooks
            "protobuf<=3.20.1",  # https://exerror.com/typeerror-descriptors-cannot-not-be-created-directly/
        ],
    },
)
