# Lint as: python3
import os

from setuptools import find_packages, setup

REQUIRED_PKGS = [
    # We use numpy>=1.17 to have np.random.Generator (Dataset shuffling)
    "numpy>=1.17",
    # Backend and serialization.
    # Minimum 3.0.0 to support mix of struct and list types in parquet,
    # and batch iterators of parquet data
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
    "pathos",
    # to get metadata of optional dependencies such as torch or tensorflow
    # for Python versions that don't have it
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
    "scikit-learn",  # restricted by hatesonar pkg ==0.23.2
    "seqeval",
    "jieba",
    "apache-beam",
]

AUDIO_REQUIRE = []

BENCHMARKS_REQUIRE = []

TESTS_REQUIRE = []


QUALITY_REQUIRE = ["pre-commit"]


EXTRAS_REQUIRE = {
    "dev": TESTS_REQUIRE + QUALITY_REQUIRE,
    "tests": TESTS_REQUIRE,
    "quality": QUALITY_REQUIRE,
}

setup(
    name="datalabs",
    version="0.4.15",
    description="Datalabs",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="expressai",
    author_email="stefanpengfei@gamil.com",
    url="https://github.com/expressai/datalab",
    download_url="https://github.com/expressai/datalab/tags",
    license="Apache 2.0",
    packages=find_packages(),
    package_data={
        "datalabs": ["py.typed", "scripts/templates/*"],
        "datalabs.utils.resources": ["*.json", "*.yaml"],
        "datalabs.operations.featurize.pre_models": ["*.pkl", "*.json"],
        "datalabs.operations.featurize.resources.gender_data": ["*.json"],
        "datalabs.operations.edit.resources": ["*.json", "*.txt", "*.names", "*.tsv"],
    },
    entry_points={
        "console_scripts": ["datalabs-cli=datalabs.commands.datasets_cli:main"]
    },
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
    include_package_data=True,
)

os.system("python -m spacy download en_core_web_sm")
