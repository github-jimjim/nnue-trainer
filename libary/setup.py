from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="nnue_trainer",
    version="0.1.1",
    description="A PyTorch Lightning NNUE model for chess engines – train and quantize with ease!",
    long_description=long_description,
    long_description_content_type="text/markdown",  
    author="Jimmy Luong",
    url="https://github.com/github-jimjim/nnue-trainer",
    author_email="nguyenhungjimmy.luong@gmail.com",
    packages=find_packages(),
    install_requires=[
        "torch==2.6.0+cu126",
        "pytorch_lightning==2.5.1",
        "numpy==1.24.3",
        "python-chess==1.999",
    ],
    entry_points={
        "console_scripts": [
            "nnue_trainer=nnue_trainer.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)