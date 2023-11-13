import os
import sys
from datetime import datetime
from typing import List

from setuptools import find_packages, setup

def fetch_requirements(path) -> List[str]:
    """
    This function reads the requirements file.

    Args:
        path (str): the path to the requirements file.

    Returns:
        The lines in the requirements file.
    """
    with open(path, "r") as fd:
        return [r.strip() for r in fd.readlines()]


def fetch_readme() -> str:
    """
    This function reads the README.md file in the current directory.

    Returns:
        The lines in the README file.
    """
    with open("README.md", encoding="utf-8") as f:
        return f.read()
# use date as the nightly version
version = datetime.today().strftime("%Y.%m.%d")
package_name = "pipegoose-nightly"

setup(
    name=package_name,
    version=version,
    packages=find_packages(
        exclude=(
            "tests",
            "docs",
            "examples",
            "tests",
            "*.egg-info",
        )
    ),
    description="A library for 3d parallelism",
    long_description=fetch_readme(),
    long_description_content_type="text/markdown",
    license="MIT License",
    install_requires=fetch_requirements("requirements.txt"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Environment :: GPU :: NVIDIA CUDA",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
    ],
)