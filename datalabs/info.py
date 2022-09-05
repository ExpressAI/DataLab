# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the TensorFlow Datasets
# Authors, DataLab Authors.
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
# See the License for the specific language gfoverning permissions and
# limitations under the License.

# Lint as: python3
""" DatasetInfo and MetricInfo record information we know about a dataset and a metric.

This includes things that we know about the dataset statically, i.e.:
 - description
 - canonical location
 - does it have validation and tests splits
 - size
 - etc.

This also includes the things that can and should be computed once we've
processed the dataset as well:
 - number of examples (in each split)
 - etc.
"""

import copy
import dataclasses
from dataclasses import asdict, dataclass, field
import json
import os
import re
from typing import Any, List, Optional, Union

import pymongo

from datalabs import config
from datalabs.features import ClassLabel, Features, Value
from datalabs.prompt import Prompt
from datalabs.splits import SplitDict
from datalabs.tasks import task_template_from_dict, TaskTemplate
from datalabs.utils import Version
from datalabs.utils.logging import get_logger
from datalabs.utils.py_utils import unique_values

logger = get_logger(__name__)


@dataclass
class SupervisedKeysData:
    input: str = ""
    output: str = ""


@dataclass
class DownloadChecksumsEntryData:
    key: str = ""
    value: str = ""


class MissingCachedSizesConfigError(Exception):
    """The expected cached sizes of the download file are missing."""


class NonMatchingCachedSizesError(Exception):
    """The prepared split doesn't have expected sizes."""


@dataclass
class PostProcessedInfo:
    features: Optional[Features] = None
    resources_checksums: Optional[dict] = None

    def __post_init__(self):
        # Convert back to the correct classes when we reload from dict
        if self.features is not None and not isinstance(self.features, Features):
            self.features = Features.from_dict(self.features)

    @classmethod
    def from_dict(cls, post_processed_info_dict: dict) -> "PostProcessedInfo":
        field_names = set(f.name for f in dataclasses.fields(cls))
        return cls(
            **{k: v for k, v in post_processed_info_dict.items() if k in field_names}
        )


@dataclass
class SpeakerDemographic:
    gender: Optional[str] = None
    race: Optional[str] = None
    ethnicity: Optional[str] = None
    native_language: Optional[str] = None
    socioeconomic_status: Optional[str] = None
    number_of_different_speakers_represented: Optional[str] = None
    presence_of_disordered_speech: Optional[str] = None
    training_in_linguistics: Optional[str] = None


@dataclass
class AnnotatorDemographic:
    gender: Optional[str] = None
    race: Optional[str] = None
    ethnicity: Optional[str] = None
    native_language: Optional[str] = None
    socioeconomic_status: Optional[str] = None
    number_of_different_speakers_represented: Optional[str] = None
    presence_of_disordered_speech: Optional[str] = None
    training_in_linguistics: Optional[str] = None


@dataclass
class SpeechSituation:
    time: Optional[str] = None
    place: Optional[str] = None
    modality: Optional[str] = None
    intended_audience: Optional[str] = None


@dataclass
class SizeInfo:
    samples: Optional[int] = None
    storage: Optional[str] = None


@dataclass
class Popularity:
    number_of_download: Optional[int] = None
    number_of_times: Optional[int] = None
    number_of_reposts: Optional[int] = None
    number_of_visits: Optional[int] = None


# @dataclass
# class PromptResult:
#     setting = "zero-shot"
#     value: float = 0.0
#     plm: str = None
#     metric: str = None


"""Example
{
      "language": "en",
      "template": "{Text}, Overall it is a {Answer} movie.",
      "answer": {
        "positive": ["fantastic", "interesting"],
        "negative": ["boring"]
      },
      "supported_plm_types": ["masked_lm", "left_to_right", "encoder_decoder"],
      "results": [
        {
          "plm": "BERT",
          "metric": "accuracy",
          "setting": "zero-shot",
          "value": "87"
        },
        {
          "plm": "BART",
          "metric": "accuracy",
          "setting": "zero-shot",
          "value": "80"
        }

      ]
    }
"""


# @dataclass
# class Prompt:
#     id: str = "null"  # this will be automatically assigned
#     language: str = "en"
#     description: str = "prompt description"
#     template: str = None
#     answers: dict = None
#     supported_plm_types: List[str] = None
#     signal_type: List[str] = None
#     results: List[PromptResult] = None
#     # features:Optional[Features] = None # {"length":Value("int64"),
#     "shape":Value("string"), "skeleton": Value("string")}
#     features: Optional[dict] = None  # {"length":5, "shape":"prefix",
#     "skeleton": "what_about"}
#     reference: str = None
#     contributor: str = "Datalab"
#
#     def __post_init__(self):
#         # Convert back to the correct classes when we reload from dict
#         if self.template is not None and self.answers is not None:
#             if isinstance(self.answers, dict):
#                 self.id = hashlib.md5((self.template +
#                 json.dumps(self.answers)).encode()).hexdigest()
#             if isinstance(self.answers, str):
#                 self.id = hashlib.md5((self.template + self.answers).
#                 encode()).hexdigest()
#             else:
#                 self.id = hashlib.md5(self.template.encode()).hexdigest()


class MongoDBClientCore:
    def __init__(self, cluster: str):
        assert re.match(r"cluster[01]", cluster)

        self.cluster = cluster
        self.url = ""
        self.client = pymongo.MongoClient(self.url)


# Singleton Wrapper of __MongoDBClient
class MongoDBClient:
    clients = {}

    def __init__(self, cluster: str):
        if not self.clients.__contains__(cluster):
            self.clients[cluster] = MongoDBClientCore(cluster)
        self.core = self.clients[cluster]

    def __query(self, database: str, collection: str, query: dict, one=True):
        col = self.core.client[database][collection]
        return col.find_one(query) if one else col.find(query)

    def __insert(self, database: str, collection: str, data: dict):
        col = self.core.client[database][collection]
        col.insert_one(data)

    # DO NOT use this unless you have confirmed to delete some db/col
    def drop(self, database: str, collection: str = None, confirm: bool = False):
        if not confirm:
            return
        target = self.core.client[database]
        if collection is not None:
            target = target[collection]
        target.drop()

    def query_metadata(self, dataset_name: str):
        return self.__query(
            "metadata", "dataset_metadata", {"dataset_name": dataset_name}
        )

    def insert_metadata(self, metadata: dict):
        self.__insert("metadata", "dev_dataset_metadata", metadata)

    def insert_sample(self, collection: str, sample: dict):
        self.__insert("dev_samples_of_dataset", collection, sample)


@dataclass
class DatasetInfo:
    """Information about a dataset.

    `DatasetInfo` documents datalab, including its name, version, and features.
    See the constructor arguments and properties for a full list.

    Note: Not all fields are known on construction and may be updated later.

    Attributes:
        description (str): A description of the dataset.
        citation (str): A BibTeX citation of the dataset.
        homepage (str): A URL to the official homepage for the dataset.
        license (str): The dataset's license. It can be the name of the
        license or a paragraph containing the terms of the license.
        features (Features, optional): The features used to specify the
        dataset's column types.
        post_processed (PostProcessedInfo, optional): Information regarding
         the resources of a possible post-processing of a dataset. For example,
          it can contain the information of an index.
        supervised_keys (SupervisedKeysData, optional): Specifies the input
         feature and the label for supervised learning if applicable for the
          dataset (legacy from TFDS).
        builder_name (str, optional): The name of the :class:`GeneratorBasedBuilder`
         subclass used to create the dataset. Usually matched to the corresponding
         script name. It is also the snake_case version of the dataset builder class
          name.
        config_name (str, optional): The name of the configuration derived from
        :class:`BuilderConfig`
        version (str or Version, optional): The version of the dataset.
        splits (dict, optional): The mapping between split name and metadata.
        download_checksums (dict, optional): The mapping between the URL to
         download the dataset's checksums and corresponding metadata.
        task_templates (List[TaskTemplate], optional): The task templates
         to prepare the dataset for during training and evaluation. Each template
          casts the dataset's :class:`Features` to standardized column names and
          types as detailed in :py:mod:`datalab.tasks`.
        **config_kwargs: Keyword arguments to be passed to the :
        class:`BuilderConfig` and used in the :class:`DatasetBuilder`.
    """

    # Set in the dataset scripts
    description: str = field(default_factory=str)
    citation: str = field(default_factory=str)
    homepage: str = field(default_factory=str)
    license: str = field(default_factory=str)
    features: Optional[Features] = None
    features_dataset: Optional[Features] = None
    post_processed: Optional[PostProcessedInfo] = None
    supervised_keys: Optional[SupervisedKeysData] = None
    task_templates: Optional[List[TaskTemplate]] = None

    # Set later by the builder
    builder_name: Optional[str] = None
    config_name: Optional[str] = None
    version: Optional[Union[str, Version]] = None
    # Set later by `download_and_prepare`
    splits: Optional[dict] = None
    download_checksums: Optional[dict] = None
    # Hide these attributes
    # download_size: Optional[int] = None
    # post_processing_size: Optional[int] = None
    # dataset_size: Optional[int] = None
    # size_in_bytes: Optional[int] = None

    # Newly Added
    prompts: List[Prompt] = None

    # Needed by MongoDB
    # string
    dataset_name: Optional[str] = None
    sub_dataset: Optional[str] = None
    homepage: Optional[str] = None
    repository: Optional[str] = None
    leaderboard: Optional[str] = None
    person_of_contact: Optional[str] = None
    production_status: Optional[str] = None
    huggingface_link: Optional[str] = None
    curation_rationale: Optional[str] = None
    genre: Optional[str] = None
    quality: Optional[str] = None
    similar_datasets: Optional[str] = None
    creator_id: Optional[str] = None
    submitter_id: Optional[str] = None
    multilinguality: Optional[str] = None
    transformation: Optional[str] = None
    # other type
    languages: Optional[List] = field(default_factory=list)
    model_ids: Optional[list] = field(default_factory=list)
    speaker_demographic: Optional[SpeakerDemographic] = field(
        default_factory=SpeakerDemographic
    )
    annotator_demographic: Optional[AnnotatorDemographic] = field(
        default_factory=AnnotatorDemographic
    )
    speech_situation: Optional[SpeechSituation] = field(default_factory=SpeechSituation)
    size: Optional[SizeInfo] = field(default_factory=SizeInfo)
    popularity: Optional[Popularity] = field(default_factory=Popularity)

    def __post_init__(self):
        # Convert back to the correct classes when we reload from dict
        if self.features is not None and not isinstance(self.features, Features):
            self.features = Features.from_dict(self.features)
        if self.post_processed is not None and not isinstance(
            self.post_processed, PostProcessedInfo
        ):
            self.post_processed = PostProcessedInfo.from_dict(self.post_processed)
        if self.version is not None and not isinstance(self.version, Version):
            if isinstance(self.version, str):
                self.version = Version(self.version)
            else:
                self.version = Version.from_dict(self.version)
        if self.splits is not None and not isinstance(self.splits, SplitDict):
            self.splits = SplitDict.from_split_dict(self.splits)
        if self.supervised_keys is not None and not isinstance(
            self.supervised_keys, SupervisedKeysData
        ):
            if isinstance(self.supervised_keys, (tuple, list)):
                self.supervised_keys = SupervisedKeysData(*self.supervised_keys)
            else:
                self.supervised_keys = SupervisedKeysData(**self.supervised_keys)

        # Parse and make a list of templates
        if self.task_templates is not None:
            if isinstance(self.task_templates, (list, tuple)):
                templates = [
                    task_template_from_dict(template)
                    if isinstance(template, dict)
                    else template
                    for template in self.task_templates
                ]
                self.task_templates = [
                    template for template in templates if template is not None
                ]
            elif isinstance(self.task_templates, TaskTemplate):
                self.task_templates = [self.task_templates]
            else:
                template = task_template_from_dict(self.task_templates)
                self.task_templates = [template] if template is not None else []

        # Insert labels and mappings for text classification
        if self.task_templates is not None:
            self.task_templates = list(self.task_templates)
            if self.features is not None:
                for idx, template in enumerate(self.task_templates):

                    if template.label_schema is None:
                        continue

                    is_label_column = None
                    for col_name, type_name in template.label_schema.items():
                        # for sequence labeling
                        if col_name == "tags":
                            is_label_column = col_name
                        elif type_name == ClassLabel:
                            is_label_column = col_name
                    if is_label_column is not None and isinstance(
                        self.features[is_label_column], ClassLabel
                    ):
                        labels = self.features[template.label_column].names
                        self.task_templates[idx].set_labels(labels)
                    elif is_label_column is not None and hasattr(
                        template, "tags_column"
                    ):
                        # for sequence labeling tasks
                        labels = self.features[template.tags_column].feature.names
                        self.task_templates[idx].set_labels(labels)

        # Protected from other codes:
        self.download_size = None
        self.post_processing_size = None
        self.dataset_size = None
        self.size_in_bytes = None
        # self._init_db_attr()

    def _init_db_attr(self):
        metadata = MongoDBClient("cluster0").query_metadata(self.builder_name)
        if type(metadata) != dict:
            # The dataset may not be uploaded yet
            self._infer_attr()
        else:
            self._fill_db_attr(metadata)

    def _infer_attr(self):
        if self.dataset_name is None:
            self.dataset_name = self.builder_name

    def _fill_db_attr(self, metadata):
        self._fill_single_attr(metadata, "dataset_name")
        self._fill_single_attr(metadata, "sub_dataset")
        self._fill_single_attr(metadata, "homepage")
        self._fill_single_attr(metadata, "repository")
        self._fill_single_attr(metadata, "leaderboard")
        self._fill_single_attr(metadata, "person_of_contact")
        self._fill_single_attr(metadata, "production_status")
        self._fill_single_attr(metadata, "huggingface_link")
        self._fill_single_attr(metadata, "curation_rationale")
        self._fill_single_attr(metadata, "genre")
        self._fill_single_attr(metadata, "quality")
        self._fill_single_attr(metadata, "similar_datasets")
        self._fill_single_attr(metadata, "creator_id")
        self._fill_single_attr(metadata, "submitter_id")
        self._fill_single_attr(metadata, "Multilinguality")
        self._fill_single_attr(metadata, "operations")

        self._fill_single_attr(metadata, "languages")
        self._fill_single_attr(metadata, "model_ids")
        self._fill_single_attr(metadata, "speaker_demographic")
        self._fill_single_attr(metadata, "annotator_demographic")
        self._fill_single_attr(metadata, "speech_situation")
        self._fill_single_attr(metadata, "size")
        self._fill_single_attr(metadata, "popularity")

    def _fill_single_attr(self, src: dict, attr: str):
        if src.__contains__(attr):
            self.__dict__[attr] = src[attr]

    def _license_path(self, dataset_info_dir):
        return os.path.join(dataset_info_dir, config.LICENSE_FILENAME)

    def write_to_directory(self, dataset_info_dir):
        """Write `DatasetInfo` as JSON to `dataset_info_dir`.

        Also save the license separately in LICENCE.
        """
        with open(
            os.path.join(dataset_info_dir, config.DATASET_INFO_FILENAME), "wb"
        ) as f:
            self._dump_info(f)

        with open(os.path.join(dataset_info_dir, config.LICENSE_FILENAME), "wb") as f:
            self._dump_license(f)

    def _as_dict(self):
        return asdict(self)

    def _dump_info(self, file):
        """Dump info in `file` file-like object open in bytes mode (to support
        remote files)"""
        file.write(json.dumps(asdict(self)).encode("utf-8"))

    def _dump_license(self, file):
        """Dump license in `file` file-like object open in bytes mode (to
        support remote files)"""
        file.write(self.license.encode("utf-8"))

    @classmethod
    def from_merge(cls, dataset_infos: List["DatasetInfo"]):
        dataset_infos = [
            dset_info.copy() for dset_info in dataset_infos if dset_info is not None
        ]
        description = "\n\n".join(
            unique_values(info.description for info in dataset_infos)
        )
        citation = "\n\n".join(unique_values(info.citation for info in dataset_infos))
        homepage = "\n\n".join(unique_values(info.homepage for info in dataset_infos))
        license = "\n\n".join(unique_values(info.license for info in dataset_infos))
        features = None
        supervised_keys = None
        task_templates = None

        # Find common task templates across all dataset infos
        all_task_templates = [
            info.task_templates
            for info in dataset_infos
            if info.task_templates is not None
        ]
        if len(all_task_templates) > 1:
            task_templates = list(
                set(all_task_templates[0]).intersection(*all_task_templates[1:])
            )
        elif len(all_task_templates):
            task_templates = list(set(all_task_templates[0]))
        # If no common task templates found, replace empty list with None
        task_templates = task_templates if task_templates else None

        return cls(
            description=description,
            citation=citation,
            homepage=homepage,
            license=license,
            features=features,
            supervised_keys=supervised_keys,
            task_templates=task_templates,
        )

    @classmethod
    def from_directory(cls, dataset_info_dir: str) -> "DatasetInfo":
        """Create DatasetInfo from the JSON file in `dataset_info_dir`.

        This function updates all the dynamically generated fields (num_examples,
        hash, time of creation,...) of the DatasetInfo.

        This will overwrite all previous metadata.

        Args:
            dataset_info_dir (`str`): The directory containing the metadata file. This
                should be the root directory of a specific dataset version.
        """
        logger.info(f"Loading Dataset info from {dataset_info_dir}")
        if not dataset_info_dir:
            raise ValueError(
                "Calling DatasetInfo.from_directory() with undefined dataset_info_dir."
            )

        with open(
            os.path.join(dataset_info_dir, config.DATASET_INFO_FILENAME),
            "r",
            encoding="utf-8",
        ) as f:
            dataset_info_dict = json.load(f)
        return cls.from_dict(dataset_info_dict)

    @classmethod
    def from_dict(cls, dataset_info_dict: dict) -> "DatasetInfo":
        field_names = set(f.name for f in dataclasses.fields(cls))
        return cls(**{k: v for k, v in dataset_info_dict.items() if k in field_names})

    def update(self, other_dataset_info: "DatasetInfo", ignore_none=True):
        self_dict = self.__dict__
        self_dict.update(
            **{
                k: copy.deepcopy(v)
                for k, v in other_dataset_info.__dict__.items()
                if (v is not None or not ignore_none)
            }
        )

    def copy(self) -> "DatasetInfo":
        protect_list = [
            "download_size",
            "post_processing_size",
            "dataset_size",
            "size_in_bytes",
        ]
        argv = filter(lambda item: item[0] not in protect_list, self.__dict__.items())
        return self.__class__(**{k: copy.deepcopy(v) for k, v in argv})


class DatasetInfosDict(dict):
    def write_to_directory(self, dataset_infos_dir, overwrite=False):
        total_dataset_infos = {}
        dataset_infos_path = os.path.join(
            dataset_infos_dir, config.DATASETDICT_INFOS_FILENAME
        )
        if os.path.exists(dataset_infos_path) and not overwrite:
            logger.info(
                f"Dataset Infos already exists in {dataset_infos_dir}. Completing"
                f" it with new infos."
            )
            total_dataset_infos = self.from_directory(dataset_infos_dir)
        else:
            logger.info(f"Writing new Dataset Infos in {dataset_infos_dir}")
        total_dataset_infos.update(self)
        with open(dataset_infos_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    config_name: asdict(dset_info)
                    for config_name, dset_info in total_dataset_infos.items()
                },
                f,
            )

    @classmethod
    def from_directory(cls, dataset_infos_dir):
        logger.info(f"Loading Dataset Infos from {dataset_infos_dir}")
        with open(
            os.path.join(dataset_infos_dir, config.DATASETDICT_INFOS_FILENAME),
            "r",
            encoding="utf-8",
        ) as f:
            dataset_infos_dict = {
                config_name: DatasetInfo.from_dict(dataset_info_dict)
                for config_name, dataset_info_dict in json.load(f).items()
            }
        return cls(**dataset_infos_dict)


@dataclass
class MetricInfo:
    """Information about a metric.

    `MetricInfo` documents a metric, including its name, version, and features.
    See the constructor arguments and properties for a full list.

    Note: Not all fields are known on construction and may be updated later.
    """

    # Set in the dataset scripts
    description: str
    citation: str
    features: Features
    inputs_description: str = field(default_factory=str)
    homepage: str = field(default_factory=str)
    license: str = field(default_factory=str)
    codebase_urls: List[str] = field(default_factory=list)
    reference_urls: List[str] = field(default_factory=list)
    streamable: bool = False
    format: Optional[str] = None

    # Set later by the builder
    metric_name: Optional[str] = None
    config_name: Optional[str] = None
    experiment_id: Optional[str] = None

    def __post_init__(self):
        if "predictions" not in self.features:
            raise ValueError(
                "Need to have at least a 'predictions' field in 'features'."
            )
        if self.format is not None:
            for key, value in self.features.items():
                if not isinstance(value, Value):
                    raise ValueError(
                        f"When using 'numpy' format, all features should be"
                        f" a `datalab.Value` feature. "
                        f"Here {key} is an instance of {value.__class__.__name__}"
                    )

    def write_to_directory(self, metric_info_dir):
        """Write `MetricInfo` as JSON to `metric_info_dir`.
        Also save the license separately in LICENCE.
        """
        with open(
            os.path.join(metric_info_dir, config.METRIC_INFO_FILENAME),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(asdict(self), f)

        with open(
            os.path.join(metric_info_dir, config.LICENSE_FILENAME),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(self.license)

    @classmethod
    def from_directory(cls, metric_info_dir) -> "MetricInfo":
        """Create MetricInfo from the JSON file in `metric_info_dir`.

        Args:
            metric_info_dir: `str` The directory containing the metadata file. This
                should be the root directory of a specific dataset version.
        """
        logger.info(f"Loading Metric info from {metric_info_dir}")
        if not metric_info_dir:
            raise ValueError(
                "Calling MetricInfo.from_directory() with undefined metric_info_dir."
            )

        with open(
            os.path.join(metric_info_dir, config.METRIC_INFO_FILENAME),
            "r",
            encoding="utf-8",
        ) as f:
            metric_info_dict = json.load(f)
        return cls.from_dict(metric_info_dict)

    @classmethod
    def from_dict(cls, metric_info_dict: dict) -> "MetricInfo":
        field_names = set(f.name for f in dataclasses.fields(cls))
        return cls(**{k: v for k, v in metric_info_dict.items() if k in field_names})


"""
The following is introduced for explainaboard
"""


@dataclass
class Table:
    # def __init__(self,
    #              table_iterator):
    #     self.table = []
    #     for _id, dict_features in table_iterator:
    #         self.table.append(dict_features)
    table: dict = None

    # def __post_init__(self):


@dataclass
class PaperInfo:
    """
    "year": "xx",
    "venue": "xx",
    "title": "xx",
    "author": "xx",
    "url": "xx",
    "bib": "xx"
    """

    year: Optional[str] = None
    venue: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None
    url: Optional[str] = None
    bib: Optional[str] = None


@dataclass
class Performance:
    metric_name: float = None
    value: float = None
    confidence_score_low: float = None
    confidence_score_up: float = None


@dataclass
class BucketPerformance(Performance):
    bucket_name: str = None
    n_samples: float = None
    bucket_samples: Any = None


@dataclass
class Result:
    overall: Any = None
    calibration: List[Performance] = None
    fine_grained: Any = None
    is_print_case: bool = True
    is_print_confidence_interval: bool = True


@dataclass
class SysOutputInfo:
    """Information about a system output

    Attributes:
        model_name (str): the name of the system .
        dataset_name (str): the dataset used of the system.
        language (str): the language of the dataset.
        code (str): the url of the code.
        download_link (str): the url of the system output.
        paper (Paper, optional): the published paper of the system.
        features (Features, optional): the features used to describe system output's
                                        column type.
    """

    # set in the system_output scripts
    task_name: str
    model_name: Optional[str] = None
    dataset_name: Optional[str] = None
    sub_dataset_name: Optional[str] = None
    metric_names: Optional[List[str]] = None
    reload_stat: bool = True
    # language : str = "English"

    # set later
    # code: str = None
    # download_link: str = None
    # paper_info: PaperInfo = PaperInfo()
    features: Features = None
    results: Result = field(default_factory=lambda: Result())

    def write_to_directory(self, dataset_info_dir):
        """Write `SysOutputInfo` as JSON to `dataset_info_dir`."""
        with open(
            os.path.join(dataset_info_dir, config.SYS_OUTPUT_INFO_FILENAME), "wb"
        ) as f:
            self._dump_info(f)

    def to_dict(self) -> dict:
        return asdict(self)

    def print_as_json(self):
        print(json.dumps(self.to_dict(), indent=4))

    def _dump_info(self, file):
        """SystemOutputInfo => JSON"""
        file.write(json.dumps(self.to_dict(), indent=4).encode("utf-8"))

    @classmethod
    def from_directory(cls, sys_output_info_dir: str) -> "SysOutputInfo":
        """Create SysOutputInfo from the JSON file in `sys_output_info_dir`.
        Args:
            sys_output_info_dir (`str`): The directory containing the metadata
            file. This
                should be the root directory of a specific dataset version.
        """
        logger.info("Loading Dataset info from %s", sys_output_info_dir)
        if not sys_output_info_dir:
            raise ValueError(
                "Calling DatasetInfo.from_directory() with undefined dataset_info_dir."
            )

        with open(
            os.path.join(sys_output_info_dir, config.SYS_OUTPUT_INFO_FILENAME),
            "r",
            encoding="utf-8",
        ) as f:
            sys_output_info_dict = json.load(f)
        return cls.from_dict(sys_output_info_dict)

    # @classmethod
    # def from_dict(cls, task_name: str, sys_output_info_dict: dict) -> "SysOutputInfo":
    #     field_names = set(f.name for f in dataclasses.fields(cls))
    #     return cls(task_name, **{k: v for k, v in sys_output_info_dict.items()
    #     if k in field_names})

    @classmethod
    def from_dict(cls, sys_output_info_dict: dict) -> "SysOutputInfo":
        field_names = set(f.name for f in dataclasses.fields(cls))
        return cls(
            **{k: v for k, v in sys_output_info_dict.items() if k in field_names}
        )

    def update(self, other_sys_output_info: "SysOutputInfo", ignore_none=True):
        self_dict = self.__dict__
        self_dict.update(
            **{
                k: copy.deepcopy(v)
                for k, v in other_sys_output_info.__dict__.items()
                if (v is not None or not ignore_none)
            }
        )

    def copy(self) -> "SysOutputInfo":
        return self.__class__(**{k: copy.deepcopy(v) for k, v in self.__dict__.items()})
