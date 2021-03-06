#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

viz_requirements = [
    "simulariumio>=1.4.0",
]

setup_requirements = [
    "pytest-runner>=5.2",
]

test_requirements = [
    "black>=19.10b0",
    "codecov>=2.1.4",
    "flake8>=3.8.3",
    "flake8-debugger>=3.2.1",
    "pytest>=5.4.3",
    "pytest-cov>=2.9.0",
    "pytest-raises>=0.11",
    *viz_requirements,
]

dev_requirements = [
    *setup_requirements,
    *test_requirements,
    "bump2version>=1.0.1",
    "coverage>=5.1",
    "ipython>=7.15.0",
    "m2r2>=0.2.7",
    "pytest-runner>=5.2",
    "Sphinx>=3.4.3",
    "sphinx_rtd_theme>=0.5.1",
    "tox>=3.15.2",
    "twine>=3.1.1",
    "wheel>=0.34.2",
]

example_requirements = [
    "psutil",
    "awscli",
    "boto3",
    "openpyxl>=3.0",
    *viz_requirements,
]

requirements = [
    "numpy>=1.16",
    "scipy>=1.5.2",
    "pandas>=1.0",
]

extra_requirements = {
    "setup": setup_requirements,
    "test": test_requirements,
    "dev": dev_requirements,
    "viz": viz_requirements,
    "examples": example_requirements,
    "all": [*requirements, *dev_requirements, *viz_requirements, *example_requirements],
}

setup(
    author="Blair Lyons",
    author_email="blairl@alleninstitute.org",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: Free for non-commercial use",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    description="Tools for building computational biology models and example models from the Simularium project.",
    entry_points={
        "console_scripts": ["my_example=simularium_models_util.bin.my_example:main"],
    },
    install_requires=requirements,
    license="Allen Institute Software License",
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="simularium_models_util",
    name="simularium_models_util",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*"]),
    python_requires=">=3.7",
    setup_requires=setup_requirements,
    test_suite="simularium_models_util/tests",
    tests_require=test_requirements,
    extras_require=extra_requirements,
    url="https://github.com/allen-cell-animated/simularium_models_util",
    # Do not edit this string manually, always use bumpversion
    # Details in CONTRIBUTING.rst
    version="0.0.0",
    zip_safe=False,
)
