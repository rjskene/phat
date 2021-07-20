import pathlib
from setuptools import setup

setup(
    name="phat-tails",
    version="0.0.1",
    description="Pareto Hybrids with Fat Tails",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/rskene/phat",
    author="Ryan Skene",
    author_email="rjskene83@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["reader"],
    include_package_data=True,
    install_requires=["feedparser", "html2text"],
    entry_points={
        "console_scripts": [
            "realpython=reader.__main__:main",
        ]
    },
)
