"""The setup file"""

from setuptools import setup, find_packages

setup(
    name="outliergpt",
    version="0.1.0",
    author="Devminda Abeynayake",
    author_email="devmindaabeynayake@gmail.com",
    description="A package for explaining outliers in datasets using LLMs",
    packages=find_packages(),
    install_packages=[
        "pandas",
        "requests",
        "openai",
        "scipy",
        "numpy",
    ],
)
