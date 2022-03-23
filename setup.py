# Lint as: python3


import datetime
import itertools
import os
import sys

from setuptools import find_packages, setup


REQUIRED_PKGS = [
    # We use numpy>=1.17 to have np.random.Generator (Dataset shuffling)
    "numpy>=1.17",
    # Backend and serialization.
    # Minimum 3.0.0 to support mix of struct and list types in parquet, and batch iterators of parquet data
    "pyarrow>=3.0.0,!=4.0.0",
    # For smart caching dataset processing
    "dill",
    # For performance gains with apache arrow
    "pandas",
    # for downloading datalabs over HTTPS
    "requests>=2.19.0",
    # progress bars in download and scripts
    "tqdm>=4.62.1",
    # dataclasses for Python versions that don't have it
    "dataclasses;python_version<'3.7'",
    # for fast hashing
    "xxhash",
    # for better multiprocessing
    "multiprocess",
    # to get metadata of optional dependencies such as torch or tensorflow for Python versions that don't have it
    "importlib_metadata;python_version<'3.8'",
    # to save datalabs locally or on any filesystem
    # minimum 2021.05.0 to have the AbstractArchiveFileSystem
    "fsspec[http]>=2021.05.0",
    # for data streaming via http
    "aiohttp",
    "huggingface_hub>=0.1.0,<1.0.0",
    # Utilities from PyPA to e.g., compare versions
    "packaging",
    # New dependencies needed by datalabs
    "pymongo[srv]",
    "spacy",
    "checklist",
    "lexicalrichness",
    "sacrebleu",
    "compare_mt",
    "scikit-learn", # restricted by hatesonar pkg ==0.23.2
    "py7zr",
    # for hate speech
    "hatesonar",
    "dateparser",
    "seqeval",
    "torch", # too larger
    # "explainaboard",
]

AUDIO_REQUIRE = [
    "librosa",
]

BENCHMARKS_REQUIRE = [
    "numpy==1.18.5",
    "tensorflow==2.3.0",
    "torch==1.6.0",
    "transformers==3.0.2",
]

TESTS_REQUIRE = [
    # test dependencies
    "absl-py",
    "pytest",
    "pytest-datadir",
    "pytest-xdist",
    # optional dependencies
    "apache-beam>=2.26.0",
    "elasticsearch",
    "aiobotocore",
    "boto3",
    "botocore",
    "faiss-cpu>=1.6.4",
    "fsspec[s3]",
    "moto[s3,server]==2.0.4",
    "rarfile>=4.0",
    "s3fs==2021.08.1",
    "tensorflow>=2.3,!=2.6.0,!=2.6.1",
    "torch",
    "torchaudio",
    "transformers",
    # datalabs dependencies
    "bs4",
    "conllu",
    "langdetect",
    "lxml",
    "mwparserfromhell",
    "nltk",
    "openpyxl",
    "py7zr",
    "tldextract",
    "zstandard",
    # metrics dependencies
    "bert_score>=0.3.6",
    "rouge_score",
    "sacrebleu",
    "scipy",
    "seqeval",
    "scikit-learn",
    "jiwer",
    "sentencepiece",  # for bleurt
    # to speed up pip backtracking
    "toml>=0.10.1",
    "requests_file>=1.5.1",
    "tldextract>=3.1.0",
    "texttable>=1.6.3",
    "Werkzeug>=1.0.1",
    "six~=1.15.0",
    # metadata validation
    "importlib_resources;python_version<'3.7'",
    # new dependencies needed by datalabs
    "pymongo[srv]",
    "spacy",
    "checklist",
    "lexicalrichness",
    "sacrebleu",
    "compare_mt",
    "py7zr",
]

if os.name != "nt":
    # dependencies of unbabel-comet
    # only test if not on windows since there're issues installing fairseq on windows
    TESTS_REQUIRE.extend(
        [
            "wget>=3.2",
            "pytorch-nlp==0.5.0",
            "pytorch_lightning",
            "fastBPE==0.1.0",
            "fairseq",
        ]
    )

QUALITY_REQUIRE = ["black==21.4b0", "flake8==3.7.9", "isort>=5.0.0", "pyyaml>=5.3.1"]


EXTRAS_REQUIRE = {
    "audio": AUDIO_REQUIRE,
    "apache-beam": ["apache-beam>=2.26.0"],
    "tensorflow": ["tensorflow>=2.2.0,!=2.6.0,!=2.6.1"],
    "tensorflow_gpu": ["tensorflow-gpu>=2.2.0,!=2.6.0,!=2.6.1"],
    "torch": ["torch"],
    "s3": [
        "fsspec",
        "boto3",
        "botocore",
        "s3fs",
    ],
    "streaming": [],  # for backward compatibility
    "dev": TESTS_REQUIRE + QUALITY_REQUIRE,
    "tests": TESTS_REQUIRE,
    "quality": QUALITY_REQUIRE,
    "benchmarks": BENCHMARKS_REQUIRE,
    "docs": [
        "docutils==0.16.0",
        "recommonmark",
        "sphinx==3.1.2",
        "sphinx-markdown-tables",
        "sphinx-rtd-theme==0.4.3",
        "sphinxext-opengraph==0.4.1",
        "sphinx-copybutton",
        "fsspec<2021.9.0",
        "s3fs",
        "sphinx-panels",
        "sphinx-inline-tabs",
        "myst-parser",
        "Markdown!=3.3.5",
    ],
}

setup(
    name="datalabs",
    version="0.3.7",
    description="Datalabs",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="expressai",
    author_email="stefanpengfei@gamil.com",
    url="https://github.com/expressai/datalabs",
    download_url="https://github.com/expressai/datalabs/tags",
    license="Apache 2.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"datalabs": ["py.typed", "scripts/templates/*"],
                  "datalabs.utils.resources": ["*.json", "*.yaml"],
                  "datalabs.operations.featurize.pre_models":["*.pkl","*.json"],
                  "datalabs.operations.featurize.resources.gender_data": ["*.json"],
                  "datalabs.operations.edit.resources": ["*.json","*.txt", "*.names","*.tsv"],
                  },
    entry_points={"console_scripts": ["datalabs-cli=datalabs.commands.datasets_cli:main"]},
    install_requires=REQUIRED_PKGS,
    extras_require=EXTRAS_REQUIRE,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="dataset",
    zip_safe=False,
    include_package_data=True
)

os.system("python -m spacy download en_core_web_sm")




