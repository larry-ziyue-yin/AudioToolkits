#!/usr/bin/env python3

"""FunCodec setup script."""

import os

from distutils.version import LooseVersion
from setuptools import find_packages
from setuptools import setup


requirements = {}
install_requires = []
setup_requires = []
tests_require = []
extras_require = {}

dirname = os.path.dirname(__file__)
version_file = os.path.join(dirname, "audiotoolkits", "version.txt")
with open(version_file, "r") as f:
    version = f.read().strip()

eval_reqs_path = os.path.join(dirname, "requirements-eval.txt")
if os.path.exists(eval_reqs_path):
    with open(eval_reqs_path, "r") as f:
        extras_require["eval"] = [line.strip() for line in f if line.strip() and not line.startswith("#")]
setup(
    name="audiotoolkits",
    version=version,
    license="The MIT License",
    packages=find_packages(include=["audiotoolkits*"]),
    package_data={"audiotoolkits": ["version.txt"]},
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    extras_require=extras_require,
    python_requires=">=3.8.0",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
