# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and DataLab Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import os
import platform
from pathlib import Path
from dataclasses import dataclass
from packaging import version

from .utils.logging import get_logger


logger = get_logger(__name__)

# Datasets
S3_DATASETS_BUCKET_PREFIX = "xxx"
CLOUDFRONT_DATASETS_DISTRIB_PREFIX = "xxx"
REPO_DATASETS_URL = "https://raw.githubusercontent.com/expressai/datalab/{revision}/datasets/{path}/{name}"


# Metrics
S3_METRICS_BUCKET_PREFIX = "xx"
CLOUDFRONT_METRICS_DISTRIB_PREFIX = "xx"
REPO_METRICS_URL = "https://raw.githubusercontent.com/expressai/datasets/{revision}/metrics/{path}/{name}"

# Hub
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "xx")
HUB_DATASETS_URL = HF_ENDPOINT + "/datalab/{path}/resolve/{revision}/{name}"
HUB_DEFAULT_VERSION = "main"

PY_VERSION = version.parse(platform.python_version())

if PY_VERSION < version.parse("3.8"):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata

# General environment variables accepted values for booleans
ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({"AUTO"})


# Imports
PYARROW_VERSION = version.parse(importlib_metadata.version("pyarrow"))

USE_TF = os.environ.get("USE_TF", "AUTO").upper()
USE_TORCH = os.environ.get("USE_TORCH", "AUTO").upper()
USE_JAX = os.environ.get("USE_JAX", "AUTO").upper()

TORCH_VERSION = "N/A"
TORCH_AVAILABLE = False

if USE_TORCH in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TF not in ENV_VARS_TRUE_VALUES:
    TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
    if TORCH_AVAILABLE:
        try:
            TORCH_VERSION = version.parse(importlib_metadata.version("torch"))
            logger.info(f"PyTorch version {TORCH_VERSION} available.")
        except importlib_metadata.PackageNotFoundError:
            pass
else:
    logger.info("Disabling PyTorch because USE_TF is set")

TF_VERSION = "N/A"
TF_AVAILABLE = False

if USE_TF in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TORCH not in ENV_VARS_TRUE_VALUES:
    TF_AVAILABLE = importlib.util.find_spec("tensorflow") is not None
    if TF_AVAILABLE:
        # For the metadata, we have to look for both tensorflow and tensorflow-cpu
        for package in [
            "tensorflow",
            "tensorflow-cpu",
            "tensorflow-gpu",
            "tf-nightly",
            "tf-nightly-cpu",
            "tf-nightly-gpu",
            "intel-tensorflow",
            "tensorflow-rocm",
            "tensorflow-macos",
        ]:
            try:
                TF_VERSION = version.parse(importlib_metadata.version(package))
            except importlib_metadata.PackageNotFoundError:
                continue
            else:
                break
        else:
            TF_AVAILABLE = False
    if TF_AVAILABLE:
        if TF_VERSION.major < 2:
            logger.info(f"TensorFlow found but with version {TF_VERSION}. `datalab` requires version 2 minimum.")
            TF_AVAILABLE = False
        else:
            logger.info(f"TensorFlow version {TF_VERSION} available.")
else:
    logger.info("Disabling Tensorflow because USE_TORCH is set")


JAX_VERSION = "N/A"
JAX_AVAILABLE = False

if USE_JAX in ENV_VARS_TRUE_AND_AUTO_VALUES:
    JAX_AVAILABLE = importlib.util.find_spec("jax") is not None
    if JAX_AVAILABLE:
        try:
            JAX_VERSION = version.parse(importlib_metadata.version("jax"))
            logger.info(f"JAX version {JAX_VERSION} available.")
        except importlib_metadata.PackageNotFoundError:
            pass
else:
    logger.info("Disabling JAX because USE_JAX is set to False")


USE_BEAM = os.environ.get("USE_BEAM", "AUTO").upper()
BEAM_VERSION = "N/A"
BEAM_AVAILABLE = False
if USE_BEAM in ENV_VARS_TRUE_AND_AUTO_VALUES:
    try:
        BEAM_VERSION = version.parse(importlib_metadata.version("apache_beam"))
        BEAM_AVAILABLE = True
        logger.info(f"Apache Beam version {BEAM_VERSION} available.")
    except importlib_metadata.PackageNotFoundError:
        pass
else:
    logger.info("Disabling Apache Beam because USE_BEAM is set to False")


# Optional compression tools
RARFILE_AVAILABLE = importlib.util.find_spec("rarfile") is not None
ZSTANDARD_AVAILABLE = importlib.util.find_spec("zstandard") is not None
LZ4_AVAILABLE = importlib.util.find_spec("lz4") is not None


# Cache location
DEFAULT_XDG_CACHE_HOME = "~/.cache"
XDG_CACHE_HOME = os.getenv("XDG_CACHE_HOME", DEFAULT_XDG_CACHE_HOME)
DEFAULT_HF_CACHE_HOME = os.path.join(XDG_CACHE_HOME, "expressai")
HF_CACHE_HOME = os.path.expanduser(os.getenv("HF_HOME", DEFAULT_HF_CACHE_HOME))

DEFAULT_HF_DATASETS_CACHE = os.path.join(HF_CACHE_HOME, "datalab")
HF_DATASETS_CACHE = Path(os.getenv("HF_DATASETS_CACHE", DEFAULT_HF_DATASETS_CACHE))

DEFAULT_HF_METRICS_CACHE = os.path.join(HF_CACHE_HOME, "metrics")
HF_METRICS_CACHE = Path(os.getenv("HF_METRICS_CACHE", DEFAULT_HF_METRICS_CACHE))

DEFAULT_HF_MODULES_CACHE = os.path.join(HF_CACHE_HOME, "modules")
HF_MODULES_CACHE = Path(os.getenv("HF_MODULES_CACHE", DEFAULT_HF_MODULES_CACHE))

DOWNLOADED_DATASETS_DIR = "downloads"
DEFAULT_DOWNLOADED_DATASETS_PATH = os.path.join(HF_DATASETS_CACHE, DOWNLOADED_DATASETS_DIR)
DOWNLOADED_DATASETS_PATH = Path(os.getenv("HF_DATASETS_DOWNLOADED_DATASETS_PATH", DEFAULT_DOWNLOADED_DATASETS_PATH))

EXTRACTED_DATASETS_DIR = "extracted"
DEFAULT_EXTRACTED_DATASETS_PATH = os.path.join(DEFAULT_DOWNLOADED_DATASETS_PATH, EXTRACTED_DATASETS_DIR)
EXTRACTED_DATASETS_PATH = Path(os.getenv("HF_DATASETS_EXTRACTED_DATASETS_PATH", DEFAULT_EXTRACTED_DATASETS_PATH))

# Download count for the website
HF_UPDATE_DOWNLOAD_COUNTS = (
    os.environ.get("HF_UPDATE_DOWNLOAD_COUNTS", "AUTO").upper() in ENV_VARS_TRUE_AND_AUTO_VALUES
)

# Batch size constants. For more info, see:
# https://github.com/apache/arrow/blob/master/docs/source/cpp/arrays.rst#size-limitations-and-recommendations)
DEFAULT_MAX_BATCH_SIZE = 10_000

# Pickling tables works only for small tables (<4GiB)
# For big tables, we write them on disk instead
MAX_TABLE_NBYTES_FOR_PICKLING = 4 << 30

# Offline mode
HF_DATASETS_OFFLINE = os.environ.get("HF_DATASETS_OFFLINE", "AUTO").upper() in ENV_VARS_TRUE_VALUES

# In-memory
DEFAULT_IN_MEMORY_MAX_SIZE = 0  # Disabled
IN_MEMORY_MAX_SIZE = float(os.environ.get("HF_DATASETS_IN_MEMORY_MAX_SIZE", DEFAULT_IN_MEMORY_MAX_SIZE))

# File names
DATASET_ARROW_FILENAME = "dataset.arrow"
DATASET_INDICES_FILENAME = "indices.arrow"
DATASET_STATE_JSON_FILENAME = "state.json"
DATASET_INFO_FILENAME = "dataset_info.json"
DATASETDICT_INFOS_FILENAME = "dataset_infos.json"
LICENSE_FILENAME = "LICENSE"
METRIC_INFO_FILENAME = "metric_info.json"
DATASETDICT_JSON_FILENAME = "dataset_dict.json"

MODULE_NAME_FOR_DYNAMIC_MODULES = "datasets_modules"

MAX_DATASET_CONFIG_ID_READABLE_LENGTH = 255

# Streaming

STREAMING_READ_MAX_RETRIES = 20
STREAMING_READ_RETRY_INTERVAL = 5



"""
For explainaboard
"""
@dataclass
class BuilderConfig:
    path_output_file: str = None
    path_report: str = None
    data_file: str = None
    description: str = None
    path_test_set: str = None  # for question answering task


# File Names
SYS_OUTPUT_INFO_FILENAME = "system_analysis.json"


