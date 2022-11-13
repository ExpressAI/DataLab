# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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
"""NusaX MT dataset -- Dataset providing machine translation for Indonesian, English, and 10 Indonesian local languages"""

from typing import Dict, List, Tuple

import pandas as pd

import datalabs
from datalabs import get_task, TaskType
from datalabs.utils.download_manager import DownloadManager

from nusacrowd.utils import schemas
from nusacrowd.utils.configs import NusantaraConfig
from nusacrowd.utils.constants import (DEFAULT_NUSANTARA_VIEW_NAME,
                                       DEFAULT_SOURCE_VIEW_NAME, Tasks)

_CITATION = """\
  @article{winata2022nusax,
    title={NusaX: Multilingual Parallel Sentiment Dataset for 10 Indonesian Local Languages},
    author={Winata, Genta Indra and Aji, Alham Fikri and Cahyawijaya, Samuel and Mahendra, Rahmad and Koto, Fajri and Romadhony, Ade and Kurniawan, Kemal and Moeljadi, David and Prasojo, Radityo Eko and Fung, Pascale and others},
    journal={arXiv preprint arXiv:2205.15960},
    year={2022}
    }
"""

_DESCRIPTION = """\
  NusaX is the first-ever parallel resource for 10 low-resource languages in Indonesia.
  This dataset covers the sentiment analysis task.
"""

_HOMEPAGE = "https://github.com/IndoNLP/nusax"

_LICENSE = """
  CC BY-SA 4.0
  Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
  ShareAlike — If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.
  No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.
"""

_TRAIN_DOWNLOAD_URL = "https://raw.githubusercontent.com/IndoNLP/nusax/main/datasets/mt/train.csv"
_VALID_DOWNLOAD_URL = "https://raw.githubusercontent.com/IndoNLP/nusax/main/datasets/mt/valid.csv"
_TEST_DOWNLOAD_URL = "https://raw.githubusercontent.com/IndoNLP/nusax/main/datasets/mt/test.csv"

class NusaXConfig(datalabs.BuilderConfig):
    """BuilderConfig for NusaX Config"""

    def __init__(self, **kwargs):
        """BuilderConfig for NusaX Config.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(NusaXConfig, self).__init__(**kwargs)

def nusax_config_constructor(lang_source, lang_target, version="1.0.0"):
    """Construct NusaXaConfig with nusax_mt_{lang_source}_{lang_target} as the name format"""
    return NusaXConfig(
        name=f"{lang_source}-{lang_target}",
        version=datalabs.Version(version),
        description=f"NusaX MT for {lang_source} source language and  {lang_target} target language",
    )

LANGUAGES_MAP = {
    "ace": "acehnese",
    "ban": "balinese",
    "bjn": "banjarese",
    "bug": "buginese",
    "eng": "english",
    "ind": "indonesian",
    "jav": "javanese",
    "mad": "madurese",
    "min": "minangkabau",
    "nij": "ngaju",
    "sun": "sundanese",
    "bbc": "toba_batak",
}


class NusaXMT(datalabs.GeneratorBasedBuilder):
    """NusaX-MT is a parallel corpus for training and benchmarking machine translation models across 10 Indonesian local languages + Indonesian and English. The data is presented in csv format with 12 columns, one column for each language."""

    BUILDER_CONFIGS = (
        [nusax_config_constructor(lang1, lang2) for lang1 in LANGUAGES_MAP for lang2 in LANGUAGES_MAP if lang1 != lang2]
    )

    DEFAULT_CONFIG_NAME = "eng-ind"

    def _info(self) -> datalabs.DatasetInfo:
        lang_src, lang_tgt = self.config.name.split('-')[-2:]
        features_sample = datalabs.Features(
            {
                "translation": {
                    lang_src: datalabs.Value("string"),
                    lang_tgt: datalabs.Value("string"),
                },
            }
        )

        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            homepage=_HOMEPAGE,
            features=features_sample,
            supervised_keys=None,
            languages=[lang_src, lang_tgt],
            task_templates=[
                get_task(TaskType.machine_translation)(
                    translation_column="translation",
                    lang_sub_columns=[lang_src, lang_tgt],
                )
            ],
        )

    def _split_generators(self, dl_manager: datalabs.DownloadManager) -> List[datalabs.SplitGenerator]:
        """Returns SplitGenerators."""
        train_csv_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        validation_csv_path = dl_manager.download_and_extract(_VALID_DOWNLOAD_URL)
        test_csv_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={"filepath": train_csv_path},
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                gen_kwargs={"filepath": validation_csv_path},
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={"filepath": test_csv_path},
            ),
        ]

    def _generate_examples(self, filepath) -> Tuple[int, Dict]:
        df = pd.read_csv(filepath).reset_index()
        lang_src, lang_tgt = self.config.name.split('-')[-2:]

        for index, row in df.iterrows():
            ex = {
                'translation':{
                    lang_src: row[LANGUAGES_MAP[lang_src]],
                    lang_tgt: row[LANGUAGES_MAP[lang_tgt]]
                }
            }
            yield str(index), ex