# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the DataLab Datasets Authors.
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
from typing import List

import fsspec

from . import compression
from .hffilesystem import HfFileSystem


_has_s3fs = importlib.util.find_spec("s3fs") is not None

if _has_s3fs:
    from .s3filesystem import S3FileSystem  # noqa: F401

COMPRESSION_FILESYSTEMS: List[compression.BaseCompressedFileFileSystem] = [
    compression.Bz2FileSystem,
    compression.GzipFileSystem,
    compression.Lz4FileSystem,
    compression.XzFileSystem,
    compression.ZstdFileSystem,
]

# Register custom filesystems
for fs_class in COMPRESSION_FILESYSTEMS + [HfFileSystem]:
    fsspec.register_implementation(fs_class.protocol, fs_class)


def extract_path_from_uri(dataset_path: str) -> str:
    """
    preprocesses `dataset_path` and removes remote filesystem (e.g. removing ``s3://``)

    Args:
        dataset_path (``str``): path (e.g. ``dataset/train``) or remote uri (e.g. ``s3://my-bucket/dataset/train``) of the dataset directory
    """
    if "://" in dataset_path:
        dataset_path = dataset_path.split("://")[1]
    return dataset_path


def is_remote_filesystem(fs: fsspec.AbstractFileSystem) -> bool:
    """
    Validates if filesystem has remote protocol.

    Args:
        fs (``fsspec.spec.AbstractFileSystem``): An abstract super-class for pythonic file-systems, e.g. :code:`fsspec.filesystem(\'file\')` or :class:`datalab.filesystems.S3FileSystem`
    """
    if fs is not None and fs.protocol != "file":
        return True
    else:
        return False
