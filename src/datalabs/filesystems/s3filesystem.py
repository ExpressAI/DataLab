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

import s3fs


class S3FileSystem(s3fs.S3FileSystem):
    """
    ``datalab.filesystems.S3FileSystem`` is a subclass of `s3fs.S3FileSystem <https://s3fs.readthedocs.io/en/latest/api.html>`_, which is a known
    implementation of ``fsspec``. `Filesystem Spec (FSSPEC) <https://filesystem-spec.readthedocs.io/en/latest/?badge=latest>`_  is a project to
    unify various projects and classes to work with remote filesystems
    and file-system-like abstractions using a standard pythonic interface.

    Examples:
      Listing files from public s3 bucket.

      >>> import datalabs
      >>> s3 = datalabs.filesystems.S3FileSystem(anon=True)  # doctest: +SKIP
      >>> s3.ls('public-datalab/imdb/train')  # doctest: +SKIP
      ['dataset_info.json.json','dataset.arrow','state.json']

      Listing files from private s3 bucket using ``aws_access_key_id`` and ``aws_secret_access_key``.

      >>> import datalabs
      >>> s3 = datalabs.filesystems.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)  # doctest: +SKIP
      >>> s3.ls('my-private-datalab/imdb/train')  # doctest: +SKIP
      ['dataset_info.json.json','dataset.arrow','state.json']

      Using ``S3Filesystem`` with ``botocore.session.Session`` and custom ``aws_profile``.

      >>> import botocore
      >>> from datalabs.filesystems import S3Filesystem
      >>> s3_session = botocore.session.Session(profile_name='my_profile_name')
      >>>
      >>> s3 = S3FileSystem(session=s3_session)  # doctest: +SKIP


      Loading dataset from s3 using ``S3Filesystem`` and ``load_from_disk()``.

      >>> from datalabs import load_from_disk
      >>> from datalabs.filesystems import S3Filesystem
      >>>
      >>> s3 = S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)  # doctest: +SKIP
      >>>
      >>> dataset = load_from_disk('s3://my-private-datalab/imdb/train',fs=s3)  # doctest: +SKIP
      >>>
      >>> print(len(dataset))
      25000

      Saving dataset to s3 using ``S3Filesystem`` and ``dataset.save_to_disk()``.

      >>> from datalabs import load_dataset
      >>> from datalabs.filesystems import S3Filesystem
      >>>
      >>> dataset = load_dataset("imdb")
      >>>
      >>> s3 = S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)  # doctest: +SKIP
      >>>
      >>> dataset.save_to_disk('s3://my-private-datalab/imdb/train',fs=s3)  # doctest: +SKIP



    """

    __doc__ = s3fs.S3FileSystem.__doc__.split("Examples")[0] + __doc__

    pass
