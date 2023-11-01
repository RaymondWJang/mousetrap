from setuptools import setup, find_packages

setup(
    name="mousetrap",
    version="0.1.0",
    author="Raymond W. Chang",
    author_email="ray0815@g.ucla.edu",
    description="Library for location tracking of animals",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/raymondwjang/mousetrap",
    packages=find_packages(exclude=("tests", "docs")),
    install_requires=[
        # List of dependencies
        # 'numpy >= 1.18.1',
        # 'pandas >= 1.0.3'
    ],
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.org/classifiers/
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",  # Minimum version requirement of the package
    extras_require={
        "dev": [
            "pytest>=3.7",
            # Other development dependencies
        ],
    },
)
