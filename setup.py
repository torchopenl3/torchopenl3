import glob
import gzip
import imp
import os
import sys
from itertools import product

from setuptools import find_packages, setup

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

module_dir = "torchopenl3"
input_reprs = ["linear", "mel128", "mel256"]
content_type = ["music", "env"]
weight_files = [
    os.path.join(module_dir, "openl3_{}_{}_layer_weights".format(*tup))
    for tup in product(input_reprs, content_type)
] + list(glob.glob(os.path.join(module_dir, "*/*_no_mel_layer_pytorch_weights_*")))

base_url = "https://raw.githubusercontent.com/turian/openl3_numpy_weights/main/"
for weight_file in weight_files:
    if not os.path.isfile(weight_file):
        weight_fname = os.path.splitext(weight_file)[0]
        compressed_file = "{}.npz".format(weight_fname)
        compressed_path = os.path.join(module_dir, compressed_file)
        if not os.path.isfile(compressed_file):
            print("Downloading weight file {} ...".format(compressed_file))
            urlretrieve(base_url + compressed_file, compressed_path)
        print("Decompressing ...")
        with open(compressed_path, "rb") as source:
            with open(weight_file, "wb") as target:
                target.write(source.read())
        print("Decompression complete")
        os.remove(compressed_path)
        print("Removing compressed file")

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
    url="https://github.com/turian/torchopenl3/",
    author="Joseph Turian and Gyanendra Das",
    author_email="gyanendra.19je0343@am.iitism.ac.in",
    packages=find_packages(),
    entry_points={
        "console_scripts": ["torchopenl3=torchopenl3.cli:main"],
    },
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License v2.0",
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
        "nnAudio",
        "julius",
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
        ],
    },
    package_data={"torchopenl3": weight_files},
)
