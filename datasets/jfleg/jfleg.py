# coding=utf-8
# Copyright 2022 The TensorFlow datasets Authors and the HuggingFace datasets, DataLab Authors.
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
import json

import datalabs
from datalabs import get_task, TaskType
import os

logger = datalabs.logging.get_logger(__name__)

_CITATION = """\
@InProceedings{napoles-sakaguchi-tetreault:2017:EACLshort,
  author    = {Napoles, Courtney  and  Sakaguchi, Keisuke  and  Tetreault, Joel},
  title     = {JFLEG: A Fluency Corpus and Benchmark for Grammatical Error Correction},
  booktitle = {Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 2, Short Papers},
  month     = {April},
  year      = {2017},
  address   = {Valencia, Spain},
  publisher = {Association for Computational Linguistics},
  pages     = {229--234},
  url       = {http://www.aclweb.org/anthology/E17-2037}
}

@InProceedings{heilman-EtAl:2014:P14-2,
  author    = {Heilman, Michael  and  Cahill, Aoife  and  Madnani, Nitin  and  Lopez, Melissa  and  Mulholland, Matthew  and  Tetreault, Joel},
  title     = {Predicting Grammaticality on an Ordinal Scale},
  booktitle = {Proceedings of the 52nd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)},
  month     = {June},
  year      = {2014},
  address   = {Baltimore, Maryland},
  publisher = {Association for Computational Linguistics},
  pages     = {174--180},
  url       = {http://www.aclweb.org/anthology/P14-2029}
}
"""

_DESCRIPTION = """\
JFLEG (JHU FLuency-Extended GUG) corpus
"""
_URLS = "https://datalab-hub.s3.amazonaws.com/grammatical_error_correction/jfleg/jfleg.zip"


class JflegConfig(datalabs.BuilderConfig):
    """BuilderConfig for jfleg."""

    def __init__(self, **kwargs):
        """BuilderConfig for jfleg.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(JflegConfig, self).__init__(**kwargs)


class Jfleg(datalabs.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        JflegConfig(
            name="plain_text",
            version=datalabs.Version("2.1.0", ""),
            description="Plain text",
        ),
    ]

    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "original": datalabs.Value("string"),
                    "correct":
                        {
                            "m2": datalabs.features.Sequence(datalabs.Value("string")),
                            "sentence": datalabs.features.Sequence(datalabs.Value("string")),
                        }

                }
            ),
            supervised_keys=None,
            homepage="https://github.com/keisks/jfleg",
            citation=_CITATION,
            task_templates=[
                get_task(TaskType.grammatical_error_correction_m2)(

                )
            ],
        )

    def _split_generators(self, dl_manager):
        downloaded_files = dl_manager.download_and_extract(_URLS)

        return [
            datalabs.SplitGenerator(name=datalabs.Split.VALIDATION, gen_kwargs={
                "filepath": os.path.join(downloaded_files, f"jfleg/dev.json")}),
            datalabs.SplitGenerator(name=datalabs.Split.TEST,
                                    gen_kwargs={
                                        "filepath": os.path.join(downloaded_files, f"jfleg/test.json")}),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)
        key = 0
        with open(filepath, encoding="utf-8") as f:
            sentences_dic = json.load(f)
            for sent in sentences_dic:
                yield key, {
                    "original": sent['original'],
                    "correct": {
                        'm2': None,
                        'sentence': sent['sentence']
                    }
                }
                key += 1
