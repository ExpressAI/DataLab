# flake8: noqa
# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the
# TensorFlow Datasets Authors, DataLab Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
# pylint: enable=line-too-long
# pylint: disable=g-import-not-at-top,g-bad-import-order,wrong-import-position

__version__ = "0.4.12"

from packaging import version as _version
import pyarrow
from pyarrow import total_allocated_bytes

from datalabs.arrow_dataset import concatenate_datasets, Dataset
from datalabs.arrow_reader import ArrowReader, ReadInstruction
from datalabs.arrow_writer import ArrowWriter
from datalabs.builder import (
    ArrowBasedBuilder,
    BeamBasedBuilder,
    BuilderConfig,
    DatasetBuilder,
    GeneratorBasedBuilder,
)
from datalabs.combine import interleave_datasets
from datalabs.constants import *
from datalabs.dataset_dict import DatasetDict, IterableDatasetDict
from datalabs.enums import PLMType, PromptShape, SettingType, SignalType
from datalabs.features import (
    Array2D,
    Array3D,
    Array4D,
    Array5D,
    Audio,
    ClassLabel,
    features,
    Features,
    Sequence,
    Translation,
    TranslationVariableLanguages,
    Value,
)
from datalabs.fingerprint import is_caching_enabled, set_caching_enabled
from datalabs.info import DatasetInfo, MetricInfo, MongoDBClient
from datalabs.inspect import (
    get_dataset_config_names,
    get_dataset_infos,
    get_dataset_split_names,
    inspect_dataset,
    inspect_metric,
    list_datasets,
    list_metrics,
)
from datalabs.iterable_dataset import IterableDataset
from datalabs.keyhash import KeyHasher
from datalabs.load import (
    import_main_class,
    load_dataset,
    load_dataset_builder,
    load_from_disk,
    load_metric,
    prepare_module,
)
from datalabs.metric import Metric
from datalabs.operations import operation
from datalabs.operations.aggregate import aggregating
from datalabs.operations.data import *
from datalabs.prompt import Prompt, PromptResult, Prompts
from datalabs.splits import (
    NamedSplit,
    NamedSplitAll,
    percent,
    Split,
    SplitBase,
    SplitDict,
    SplitGenerator,
    SplitInfo,
    SubSplitInfo,
)
from datalabs.tasks import get_task, TaskType
from datalabs.utils import *

SCRIPTS_VERSION = "master" if _version.parse(__version__).is_devrelease else __version__
