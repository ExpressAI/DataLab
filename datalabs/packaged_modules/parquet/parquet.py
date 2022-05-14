# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors.
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

from dataclasses import dataclass
from typing import List, Optional

import pyarrow as pa
import pyarrow.parquet as pq

import datalabs

logger = datalabs.utils.logging.get_logger(__name__)
logger = datalabs.utils.logging.get_logger(__name__)


@dataclass
class ParquetConfig(datalabs.BuilderConfig):
    """BuilderConfig for Parquet."""

    batch_size: int = 10_000
    columns: Optional[List[str]] = None
    features: Optional[datalabs.Features] = None


class Parquet(datalabs.ArrowBasedBuilder):
    BUILDER_CONFIG_CLASS = ParquetConfig

    def _info(self):
        return datalabs.DatasetInfo(features=self.config.features)

    def _split_generators(self, dl_manager):
        """We handle string, list and dicts in datafiles"""
        if not self.config.data_files:
            raise ValueError(
                f"At least one data file must be specified,"
                f" but got data_files={self.config.data_files}"
            )
        data_files = dl_manager.download_and_extract(self.config.data_files)
        if isinstance(data_files, (str, list, tuple)):
            files = data_files
            if isinstance(files, str):
                files = [files]
            return [
                datalabs.SplitGenerator(
                    name=datalabs.Split.TRAIN, gen_kwargs={"files": files}
                )
            ]
        splits = []
        for split_name, files in data_files.items():
            if isinstance(files, str):
                files = [files]
            splits.append(
                datalabs.SplitGenerator(name=split_name, gen_kwargs={"files": files})
            )
        return splits

    def _generate_tables(self, files):
        schema = (
            pa.schema(self.config.features.type)
            if self.config.features is not None
            else None
        )
        if self.config.features is not None and self.config.columns is not None:
            if sorted([field.name for field in schema]) != sorted(self.config.columns):
                raise ValueError(
                    f"Tried to load parquet data with "
                    f"columns '{self.config.columns}' with"
                    f" mismatching features '{self.config.features}'"
                )
        for file_idx, file in enumerate(files):
            with open(file, "rb") as f:
                parquet_file = pq.ParquetFile(f)
                try:
                    for batch_idx, record_batch in enumerate(
                        parquet_file.iter_batches(
                            batch_size=self.config.batch_size,
                            columns=self.config.columns,
                        )
                    ):
                        pa_table = pa.Table.from_batches([record_batch])
                        if self.config.features is not None:
                            pa_table = pa.Table.from_arrays(
                                [pa_table[field.name] for field in schema],
                                schema=schema,
                            )
                        # Uncomment for debugging (will print the
                        # Arrow table size and elements)
                        # logger.warning(f"pa_table: {pa_table}
                        # num rows: {pa_table.num_rows}")
                        # logger.warning('\n'.join(str
                        # (pa_table.slice(i, 1).to_pydict()) for i
                        # in range(pa_table.num_rows)))
                        yield f"{file_idx}_{batch_idx}", pa_table
                except ValueError as e:
                    logger.error(
                        f"Failed to read file '{file}' with error {type(e)}: {e}"
                    )
                    raise
